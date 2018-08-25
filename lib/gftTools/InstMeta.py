from datetime import *
import pandas as pd
import numbers


class TObserver:
    def __init__(self, role, look_ahead):
        self.i_observer_type = role  # 观察者的角色信息
        self.td_look_ahead = look_ahead  # 观察者捕获到该信息在日期上提前了多久 >0 表示可能有 lookahead bias


class ColMeta:
    def __init__(self, col_name, coltype, gid_o_set, timezone, periodinterval, observers, decimaldigits, physicalunit,
                 datazoom, dispunit):
        self.col_name = col_name
        self.s_original_col_name = col_name
        self.col_type = coltype
        self.gid_o_set = gid_o_set
        self.timezone = timezone
        self.periodinterval = periodinterval
        self.observers = observers
        self.decimaldigits = decimaldigits
        self.physicalunit = physicalunit
        self.datazoom = datazoom
        self.dispunit = dispunit


# this is used as original data meta, so as default, it's look is 0
# primary_t is pointer to an col
class DataframeMeta:
    def __init__(self, primary_t, col_metas, begin_time, end_time):
        self.primary_t = primary_t
        self.col_metas = col_metas
        self.begin_time = begin_time
        self.end_time = end_time

    def lookback(self):
        return 0

    def get_primary_t_freq(self):
        if self.primary_t is not None:
            return self.primary_t.periodinterval
        if self.col_metas is not None:
            for col in self.col_metas:
                if col.periodinterval is not None:
                    return col.periodinterval
        # delta one day
        return pd.Timedelta(1, unit='d')


# data_dic is used for store dict of instrMetas, your can use this to create an tree struct
# basicly, every instruction result or orignal data resource is an data.
# so, when calculate the lookback, just use the loobback and primary_t for it by default imp
# if the input is an NULL or scalar, than, there would be no InstructionMeta, there for None is used to hold the place in list.
class InstructionMeta(DataframeMeta):
    def __init__(self, lookback, primary_t, col_metas, begin_time, end_time, data_dic):
        DataframeMeta.__init__(primary_t, col_metas, begin_time, end_time)
        self.lookback = lookback
        self.data_dic = data_dic
        self.output_ref = set()
        self.scalar = None

    def __init__(self, scalar):
        DataframeMeta.__init__(self, None, None, None, None)
        self.lookback = 0
        self.data_dic = None
        self.output_ref = set()
        self.scalar = scalar

    def lookback(self):
        return self.lookback

    def on_used_by_other(self, users_meta):
        self.output_ref.add(users_meta)

    def get_scalar_as_float(self, index):
        if isinstance(self.scalar, numbers):
            return float(self.scalar)
        if isinstance(self.scalar, str):
            if self.scalar.isdigit():
                return float(self.scalar)
        raise Exception("parameter[" + index + "] is not an number")


class SubContext:
    # required_begin_date is from the request_msg
    def __init__(self, required_begin_date, required_end_date):
        self.required_begin_date = required_begin_date
        self.required_end_date = required_end_date

    def reset_input(self, my_meta: InstructionMeta):
        self.input_datas = list()
        self.input_metas = list()
        self.my_meta = my_meta

    def add_input(self, data: InstructionMeta, meta: InstructionMeta):
        self.input_datas.append(data)
        self.input_metas.append(meta)


# all caches would be in this struct. so i can get both data and meta.
class CachedDataAndMeta:
    def __init__(self, data, meta):
        self.data = data
        self.meta = meta


def copy_and_slice(input_data, begin_date, end_date):
    return input_data


# represent the basic data requirement of one instruction.
# so each code would have it's own logic for it.
# by default ,the wavefront python framework would call get_instruction_meta() to get it.
# in each instruction, would have a code block like this.
# def get_instruction_meta():
#     return InstructionDefault()
class InstructionBase:
    # all these method begin will be called in the first routine of wavefront calculation
    
    def get_output_meta(self, input_list):  # get output meta of time InstructionMeta
        return None

    def need_append_lookback(self):
        return False

    def lookback_period_para_idx(self):
        return 0

    def get_look_back_period(self, input_list):
        return 0

    def need_move_observer_t(self):
        return False

    def move_observer_para_idx(self):
        return 0

    def forward_move_observer(self):
        return False

    def get_meta_begin_time(self, input_list, required_begin_time, lookback):
        return required_begin_time - lookback

    def get_meta_end_time(self, input_list, required_end_time):
        return required_end_time

    # all the method below will be called in the second routine  of wavefront calculation

    # override this method if you wanna slice data yourself
    # return a new list of data that would be used as the inputs of your instruction.
    def slice_data(self, context: SubContext):
        begin_date = self.get_slice_begin_date(context)
        end_date = self.get_slice_end_date(context)
        return copy_and_slice(context.input_datas, begin_date, end_date)

    # override this method if you wanna define your own logic for begin_date
    def get_slice_begin_date(self, context: SubContext):
        my_required_begin = context.required_begin_date - context.my_meta.lookback
        for meta in context.input_metas:
            if meta.begin_time is not None:
                if my_required_begin < meta.begin_time:
                    my_required_begin = meta.begin_time
        return my_required_begin + context.my_meta.lookback #here move forward the begin data with lookback period
    # override this method if you wanna define your own logic for end_date
    def get_slice_end_date(self, context: SubContext):
        my_required_end = context.required_begin_date
        for meta in context.input_metas:
            if meta.end_time is not None:
                if my_required_end > meta.end_time:
                    my_required_end = meta.end_time
        return my_required_end

def get_highest_freq(input_list):
    min_delta = pd.Timedelta(15, unit='d')
    for meta in input_list:
        if meta is not None:
            delta = meta.get_primary_t_freq()
            if delta < min_delta:
                min_delta = delta
    return min_delta


class InstructionDefault(InstructionBase):
    def __init__(self):
        self.b_need_append_lookback = False
        self.b_need_move_observer_t = False

    def __init__(self, lookback_para_idx):
        self.b_need_append_lookback = True
        self.i_lookback_period_para_idx = lookback_para_idx
        self.b_need_move_observer_t = False

    def __init__(self, lookback_para_idx, move_observer_para_idx, move_forward):
        self.b_need_append_lookback = True
        self.i_lookback_period_para_idx = lookback_para_idx
        self.b_need_move_observer_t = True
        self.i_move_observer_para_idx = move_observer_para_idx
        self.b_forward_move_observer_t = move_forward

    def get_look_back_period(self, input_list):
        if self.b_need_append_lookback:
            get_highest_freq(input_list)

    def set_lookback(self, para_idx):
        self.b_need_append_lookback = True
        self.i_lookback_period_para_idx = para_idx

    def set_move_observer(self, para_idx, move_forward):
        self.b_need_move_observer_t = False
        self.b_forward_move_observer_t = move_forward
        self.i_move_observer_para_idx = para_idx

    # the lookback of this instruction is depend only on it's input, but now related to it's inputs lookback
    # the input of input_meta_list may mixed up of scalar and InstructionMeta
    def getlookback(self, input_meta_list):
        if self.b_need_append_lookback:
            hFreq = get_highest_freq(input_meta_list)

    # see the definition of SubContext
    # see return the input
    def slice_data(self, context: SubContext):
        return None



        # the first iterator of calcuation is from
