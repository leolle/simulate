import numbers
import math
import pickle
import pandas as pd
from . import gftIO
from .proto import metaInfo_pb2 as mtproto
from .proto import instStream_pb2 as instStream
import copy
import os


class FreqInfo:
    def __init__(self, index):
        self.index = index
        self.avg_delta = (index.max() - index.min()) / index.size


precached_freq = dict()
precached_freq_list = list()


def get_freq(regular_days_gid):
    return precached_freq.get(regular_days_gid)


def set_freq(regular_days_gid, index):
    idx_info = FreqInfo(index)
    precached_freq_list.append((regular_days_gid, idx_info))
    precached_freq.__setitem__(regular_days_gid, idx_info)


def load_caches():
    script_dir = os.path.dirname(__file__)
    freq_gids = ["DAILY", "BUSINESSDAILY_SH_STK", "WEEKDAY", "WEEKLYFIRST", "WEEKLYMIDDLE", "WEEKLYLAST",
                 "MONTHLYFIRST", "MONTHLYMIDDLE", "MONTHLYLAST", "QUARTERLYFIRST", "QUARTERLYMIDDLE", "QUARTERLYLAST",
                 "SEMIANNUALFIRST", "SEMIANNUALMIDDLE", "SEMIANNUALLAST", "YEARLYFIRST", "YEARLYMIDDLE", "YEARLYLAST"]
    for gid in freq_gids:
        filename = 'data/' + gid + '.pkl'
        filename = os.path.join(script_dir, filename)
        val = gftIO.zload(filename)
        set_freq(gid, val)


load_caches()

min_t_for_freq_cal = pd.Timestamp('1996-01-01')


def get_freq_name_4_index(index: pd.DatetimeIndex):
    max_t = index.max()
    min_t = index.min()
    tomorrow = pd.to_datetime('today')
    tomorrow += pd.Timedelta('1D')

    need_slice_idx = False
    if min_t < min_t_for_freq_cal:
        min_t = min_t_for_freq_cal
        need_slice_idx = True

    if max_t > tomorrow:
        max_t = tomorrow
        need_slice_idx = True

    if need_slice_idx:
        # print("min_t({2}) type:{0}, max_t({3} type:{1}".format(str(type(min_t)), str(type(max_t)), str(min_t), str(max_t)))
        begin, end = index.slice_locs(min_t, max_t)
        index = index[begin:(end - 1)]

    # print("type[{0}]:{1}".format(str(type(index)), str(type(index.size))))
    freq_delta = (max_t - min_t) / index.size
    freq_delta += pd.Timedelta('2D')

    max_size = math.ceil(index.size * 1.25)
    min_size = math.floor(index.size * 0.8)
    # print("Delat is:{0}:[{1},{2}], size is :{3}".format(str(freq_delta),str(min_size),str(max_size), str(index.size)))
    for key, value in precached_freq_list:
        if value.avg_delta > freq_delta:
            # print("{0}'s delta is {1}, so break,".format(key, str(value.avg_delta)))
            break

        start_loc, end_loc = value.index.slice_locs(min_t, max_t)
        sliced_idx_size = end_loc - start_loc
        if (sliced_idx_size > max_size) or (sliced_idx_size < min_size):
            # print("Sliced idx size is{0} for {1}".format(str(sliced_idx_size), key))
            continue
        sliced_idx = value.index[start_loc:end_loc]
        intersection = sliced_idx.intersection(index)
        # the sliced_idx should be an superset of matrix.index.
        # print("Intersect size for[{0}] is {1}".format(key, str(intersection.size)))
        if (intersection.size >= index.size):
            return key
    return None


irregular_freq_name = 'IRREGULAR'


class Calendar:
    # index is required and name is optional.
    # if the index is irregular, j_gid and t_col_name is required
    def __init__(self, regular_days_gid, index, j_gid, t_col_name):
        if regular_days_gid is not None:
            if precached_freq.__contains__(regular_days_gid):
                if index is None:
                    index = get_freq(regular_days_gid).index
            elif irregular_freq_name != regular_days_gid:
                raise Exception("Calendar name should in precache_freq")
            else:
                index = None
        self.regular_days_gid = regular_days_gid
        self.index = index
        self.j_gid = j_gid
        self.t_column_name = t_col_name

    def get_avg_delta(self):
        if self.index is not None:
            # 86400 seconds is one day.
            return ((self.index.max() - self.index.min()) / self.index.size).total_seconds() / 86400
        else:
            return 0

    def __getstate__(self):
        ret = self.__dict__.copy()
        if self.regular_days_gid is not None:
            ret['index'] = None
        return ret

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.regular_days_gid is not None:
            self.index = get_freq(self.regular_days_gid).index

    def __eq__(self, other):

        if (self.regular_days_gid is not None) and (self.regular_days_gid == other.regular_days_gid):
            return True
        elif (self.j_gid == other.j_gid) and (self.t_column_name == other.t_column_name):
            return True
        return False

    def __str__(self):
        if self.regular_days_gid:
            return self.regular_days_gid
        else:
            if self.j_gid:
                if self.t_column_name:
                    return '{0}[{1}]'.format(self.j_gid, self.t_column_name)
                else:
                    return self.j_gid
            return "Unknown"

    def serialize_to_protobuf(self, cal, need_index_pickle=True):
        if self.regular_days_gid:
            cal.regularDaysGid = self.regular_days_gid
            if need_index_pickle and (self.regular_days_gid == irregular_freq_name) and self.index is not None:
                cal.indexPickle = pickle.dumps(self.index, -1)
        elif need_index_pickle  and self.index is not None:
            cal.indexPickle = pickle.dumps(self.index, -1)
        if self.j_gid:
            cal.jGid = self.j_gid
        if self.t_column_name:
            cal.tColumnName = self.t_column_name
        cal.avgDelta = self.get_avg_delta()
        return cal


def create_index(data, pt_col_name):
    if (pt_col_name is None):
        return None
    if isinstance(data, gftIO.GftTable):
        if data.matrix is not None:
            index = data.matrix.index
        else:  # it's coltable
            index = data.columnTab[pt_col_name].unique()
            index.sort()
            index = pd.DatetimeIndex(index).dropna()
    elif isinstance(data, pd.DatetimeIndex):
        index = data
    elif isinstance(data, pd.DataFrame):
        if gftIO.ismatrix(data):
            index = data.index
        else:
            index = data[pt_col_name].unique()
            index.sort()
            index = pd.DatetimeIndex(index).dropna()
    else:
        raise Exception("Data has no index.")
    return index


def create_calendar(data, pt_col_name, j_gid):
    index = create_index(data, pt_col_name)

    if index is None:
        regular_days_gid = None
    elif (index is data) and precached_freq.__contains__(j_gid):
        regular_days_gid = j_gid
    else:
        regular_days_gid = get_freq_name_4_index(index)
    # print("Find possible freq:{0} for j:{1}".format(str(regular_days_gid), j_gid))
    return Calendar(regular_days_gid, index, j_gid, pt_col_name)


class LookbackStep:
    def __init__(self, cal: Calendar, lookback_step: int,
                 prev_if_not_found):  # if prev_if_not_found is true, use prev value if not false, else, use the next value
        self.cal = cal
        self.lookback_step = lookback_step
        self.prev_if_not_found = prev_if_not_found
        if lookback_step > 0:
            if cal is None:
                # print("Calendar is None!")
                raise Exception("Calendar is None!")

    def need_serialize(self):
        return self.lookback_step > 0

    def can_merge(self, other):
        return self.cal.__eq__(other.cal) and self.prev_if_not_found == other.prev_if_not_found

    def get_input_begin_end_t(self, required_begin, required_end):
        if self.lookback_step > 0:
            if self.prev_if_not_found:
                begin_loc = self.cal.index.get_slice_bound(required_begin, 'right', 'loc')
                if begin_loc > 0:
                    begin_loc -= 1
            else:
                begin_loc = self.cal.index.get_slice_bound(required_begin, 'left', 'loc')

            if begin_loc > self.lookback_step:
                begin_loc -= self.lookback_step
            else:
                begin_loc = 0
            end_loc = self.cal.index.get_slice_bound(required_end, 'right', 'loc')
            return self.cal.index[begin_loc], self.cal.index[end_loc - 1]
        else:
            return required_begin, required_end

    def __str__(self):
        return '{0}({1})'.format(self.cal.__str__(), str(self.lookback_step))

    def check_validation(self):
        if self.lookback_step > 0:
            if (self.cal is None) or (self.cal.index is None):
                return False
        return True

    def serialize_to_protobuf(self, lookback_step, need_index_pickle):
        self.cal.serialize_to_protobuf(lookback_step.cal, need_index_pickle)
        lookback_step.lookbackStep = self.lookback_step
        return lookback_step


class CumLookbackStep:
    def __init__(self, lookback_step, required_begin, required_end):
        self.my_step = lookback_step
        if lookback_step is not None:
            self.input_begin_t, self.input_end_t = lookback_step.get_input_begin_end_t(required_begin, required_end)
        else:
            self.input_begin_t = required_begin
            self.input_end_t = required_end
        # print("lookback_step is:{0}, required[{1}:{2}], input:[{3}:{4}]".format(str(lookback_step.lookback_step), str(required_begin),
        #                                                                        str(required_end), str(self.input_begin_t), str(self.input_end_t)))
        self.input_step = None

        # resultMeta is of type

    def add_self_to_result_meta(self, resultMeta):
        if self.my_step is not None and self.my_step.need_serialize():
            self.my_step.serialize_to_protobuf(resultMeta.lookbackSteps.add(), False)
        if self.input_step is not None:
            self.input_step.add_self_to_result_meta(resultMeta)

    def set_input_lookback(self, max_cum_input_step):  # max_input_step is CumLookbackStep
        if self.can_merge(max_cum_input_step.my_step):
            self.my_step = LookbackStep(self.my_step.cal,
                                        self.my_step.lookback_step + max_cum_input_step.my_step.lookback_step,
                                        self.my_step.prev_if_not_found)
            self.input_step = max_cum_input_step.input_step
        else:
            self.input_step = max_cum_input_step

        if self.input_begin_t > self.input_step.input_begin_t:
            self.input_begin_t = self.input_step.input_begin_t

        if self.input_end_t < self.input_step.input_end_t:
            self.input_end_t = self.input_step.input_end_t

    def get_input_step_as_list(self, add_into_list: list):
        if self.input_step is not None:
            self.input_step.get_input_step_as_list(add_into_list)
        add_into_list.append(self.my_step)
        return add_into_list


def slice_gft_table(data: gftIO.GftTable, begin_time, end_time, pt_name):
    if data.matrix is not None:
        sliced_matrix = gftIO.slice_matrix(data.matrix, begin_time, end_time)
        return gftIO.GftTable.fromPythonMatrixWithGid(sliced_matrix, data.gid)
    if data.columnTab is not None:
        pt_name = gftIO.get_pt_name(data.columnTab, pt_name)
        sliced_col_tab = gftIO.slice_column_table(data.columnTab, begin_time, end_time, pt_name)
        return gftIO.GftTable.fromColumnTableWithGid(sliced_col_tab, data.gid)


def slice_data_return_new(data, begin_time: pd.Timestamp, end_time: pd.Timestamp, pt_name):
    if isinstance(data, dict):
        ret = dict()
        for key, value in data.items():
            ret[key] = slice_data_return_new(value, begin_time, end_time, pt_name)
        return ret
    if isinstance(data, gftIO.GftTable):
        return slice_gft_table(data, begin_time, end_time, pt_name)
    elif isinstance(data, pd.DataFrame):
        if gftIO.ismatrix(data):
            return gftIO.slice_matrix(data, begin_time, end_time)
        else:
            pt_name = gftIO.get_pt_name(data.columnTab, pt_name)
            return gftIO.slice_column_tab(data, begin_time, end_time, pt_name)
    elif isinstance(data, pd.core.indexes.datetimes.DatetimeIndex):
        return data[(data >= begin_time) & (data <= end_time)]
    return data


class TObserver:
    def __init__(self, role, look_ahead):
        self.i_observer_type = role  # 观察者的角色信息
        self.td_look_ahead = look_ahead  # 观察者捕获到该信息在日期上提前了多久 >0 表示可能有 lookahead bias


# column description
class ColDesc:
    def __init__(self, col_name, coltype, gid_o_set, timezone, periodinterval, observers, decimaldigits, physicalunit,
                 datazoom, dispunit):
        self.col_name = col_name  # str, name of the column
        self.s_original_col_name = col_name  # str, original name of the column
        self.col_type = coltype  # int :column type, 4 is O, 7 is date, 2 is double, see EN_PARAMETER_TYPE in java for details
        self.gid_o_set = gid_o_set  # str array, o set array.
        self.timezone = timezone  # int ,8 if not presented.
        self.periodinterval = periodinterval  # float average period interval, use this to get freqency.
        self.observers = observers  # TObserver, not used yet.
        self.decimaldigits = decimaldigits  # int, decimal digit number
        self.physicalunit = physicalunit  # str, unit
        self.datazoom = datazoom  # float data zoom.
        self.dispunit = dispunit  # str unit for display.

    @classmethod
    def create_from_pb_col_meta(cls, pb_col_meta: mtproto.colMeta):
        col_name = pb_col_meta.columnName
        col_type = pb_col_meta.colType
        if pb_col_meta.gidOsets.__len__() > 0:
            gid_o_set = list()
            for oSet in pb_col_meta.gidOsets:
                gid_o_set.append(oSet)
        else:
            gid_o_set = None

        if pb_col_meta.HasField("timezone"):
            timezone = pb_col_meta.timezone
        else:
            timezone = None

        if pb_col_meta.HasField("periodInterval"):
            periodinterval = pb_col_meta.periodInterval
        else:
            periodinterval = None

        if pb_col_meta.observers.__len__() > 0:
            observers = list()
            for obs in observers:
                observers.append(obs)
        else:
            observers = None

        if pb_col_meta.HasField("decimalDigits"):
            decimal_digits = pb_col_meta.decimalDigits
        else:
            decimal_digits = None

        if pb_col_meta.HasField("physicalUnit"):
            physicalunit = pb_col_meta.physicalUnit
        else:
            physicalunit = None

        if pb_col_meta.HasField("dataZoom"):
            datazoom = pb_col_meta.dataZoom
        else:
            datazoom = None

        if pb_col_meta.HasField("dispUnit"):
            disp_unit = pb_col_meta.dispUnit
        else:
            disp_unit = None

        return ColDesc(col_name, col_type, gid_o_set, timezone, periodinterval, observers, decimal_digits, physicalunit,
                       datazoom, disp_unit)

    def as_dict(self):
        return self.__dict__


sys_max_timestamp = gftIO.gs_day_2_pd_dt(150000)
sys_min_timestamp = gftIO.gs_day_2_pd_dt(10000)


# this is used as original data meta, so as default, it's look is 0
# primary_t is pointer to an col
# use max / min for data range
# use begin / end for data calculation period
# dataframe description
class DataframeDesc:
    def set_data(self, j_gid, primary_t, trade_calendar, col_metas, min_t, max_t, min_j_gid, max_j_gid):
        self.j_gid = j_gid  # str, gid of the data
        self.primary_t = primary_t  # ColDesc, primary_t,
        self.trade_calendar = trade_calendar
        self.col_metas = col_metas  # list of ColDesc, all columns meta.
        if min_t is not None:
            self.min_t = min_t  # min t of primary t
        else:
            self.min_t = None
        if max_t is not None:
            self.max_t = max_t  # max t of primary t
        else:
            self.max_t = None
        self.min_j_gid = min_j_gid  # gid of j which has the min t.
        self.max_j_gid = max_j_gid  # gid of j which has the max t.
        return

    def has_primary_t_and_col_metas(self):  # as the name said.
        if (self.primary_t is not None) and (self.col_metas is not None):
            return True
        return False


def find_col_with_name(col_name, col_list):
    for col in col_list:
        if col.col_name == col_name:
            return col
    return None


def getMetaId(meta):
    inst_id = getattr(meta, 'inst_id', None)
    if inst_id is not None:
        return " id:" + str(meta.inst_id)
    else:
        return " id:" + str(id(meta))


class SliceCache:
    def __init__(self, required_begin, required_end, cum_lookback):
        self.required_begin = required_begin
        self.required_end = required_end
        self.cum_lookback = cum_lookback

    def fit_require(self, required_begin, required_end):
        return (required_begin == self.required_begin) and (required_end == self.required_end)


invalid_lookback_step = LookbackStep(None, 0, False)


# Basicly, every instruction result or orignal data resource is an data.
# so, when calculate the input_lookback, just use the loobback and primary_t for it by default imp
# if the input is an NULL than, there would be no InstructionMeta, there for None is used to hold the place in list.
class InstrInstanceDesc(DataframeDesc):
    # use this as __init__()
    def set_data(self, j_gid, input_lookback: LookbackStep, primary_t, col_metas, min_t, max_t, min_j_gid, max_j_gid,
                 input_list, inst_id):
        if input_lookback:
            cal = input_lookback.cal
        else:
            cal = None
        DataframeDesc.set_data(self, j_gid, primary_t, cal, col_metas, min_t, max_t, min_j_gid, max_j_gid)
        # print("(0)InstrInstanceDesc[set_data] input_lookback:" + str(input_lookback) + " " + getMetaId(self))

        if col_metas is None:
            col_meta_size = -1
        else:
            col_meta_size = len(col_metas)
        # print("(1)InstrInstanceDesc[{1}]column meta size:{0}".format(str(col_meta_size),str(inst_id)))
        if primary_t is not  None:
            # print("have primary t of name:" + primary_t.col_name)
            for col in col_metas:
                if col.col_name == col.col_name:
                    # print("Find primary t!")
                    break;
        # input_lookback period for the instruction. if it's an original data, then input_lookback will set to 0
        self.input_lookback = input_lookback
        self.input_list = input_list  # just init it here...
        self.scalar = None
        self.place_holder = None
        self.inst_id = inst_id
        self.input_begin_t = sys_max_timestamp
        self.input_end_t = sys_min_timestamp

        if input_list:
            self.required_begin_t = sys_max_timestamp
            self.required_end_t = sys_min_timestamp
        else:
            if min_t:
                self.required_begin_t = gftIO.gs_day_2_pd_dt(min_t)
            else:
                self.required_begin_t = sys_max_timestamp
            if max_t:
                self.required_end_t = gftIO.gs_day_2_pd_dt(max_t)
            else:
                self.required_end_t = sys_min_timestamp

        self.min_t = min_t  # actually it the max of all input's min_t
        self.max_t = max_t  # and this is the min of all input's max_t
        self.min_j_gid = min_j_gid
        self.max_j_gid = max_j_gid
        self.cum_lookback_cache = None


    def set_cache_required_begin_end_t_and_ret_all_t_info(self, old_meta, user_default_input_lookback):
        no_new_data = False
        if old_meta is None:
            self.required_begin_t = gftIO.gs_day_2_pd_dt(self.min_t)
            self.required_end_t = gftIO.gs_day_2_pd_dt(self.max_t)
        else:
            self.required_begin_t = gftIO.gs_day_2_pd_dt(old_meta.required_end_t - user_default_input_lookback)
            self.required_end_t = gftIO.gs_day_2_pd_dt(self.max_t)
            no_new_data = (old_meta.required_end_t == self.max_t)
        return self.required_begin_t, self.required_end_t, self.get_slice_begin_t(), self.get_slice_end_t(), no_new_data

    def check_lookback(self):
        if self.input_lookback:
            if not self.input_lookback.check_validation():
                pass
                # print("Input_lookback invalid for:" + getMetaId(self))

    def set_required_begin_end_t(self, required_begin_t, required_end_t):  # the the cum_lookback
        # print(
        #     "(0)InstrInstanceDesc[set_required_begin_end_t] Required[{0}:{1}], available[{2}:{3} for:{4}".format(str(required_begin_t), str(required_end_t),
        #                                                           str(self.min_t), str(self.max_t), getMetaId(self)))

        if self.cum_lookback_cache and self.cum_lookback_cache.fit_require(required_begin_t, required_end_t):
            return self.cum_lookback_cache.cum_lookback

        if self.required_begin_t > required_begin_t:
            self.required_begin_t = required_begin_t

        if self.required_end_t < required_end_t:
            self.required_end_t = required_end_t


        # print("(1)InstrInstanceDesc[set_required_begin_end_t] Set required begin/end t for{0}:[{1}:{2}]".format(getMetaId(self),str(required_begin_t),str(required_end_t)))
        cum_lookback = CumLookbackStep(self.input_lookback, required_begin_t, required_end_t)
        max_lookback = None

        if self.input_list is not None:
            # print("(2)InstrInstanceDesc[set_required_begin_end_t] {0} has input list of size:{1}".format(getMetaId(self), str(len(self.input_list))))
            for input in self.input_list:
                if input is not None and input.input_list:
                    if (input.input_lookback is not None):  # means input_list is not none and has elements(means it's not original j), and it not lambda
                        input_cum_lookback = input.set_required_begin_end_t(cum_lookback.input_begin_t,
                                                                            cum_lookback.input_end_t)
                        if max_lookback and max_lookback.input_begin_t > input_cum_lookback.input_begin_t:
                            max_lookback = input_cum_lookback
                #         else:
                #             print("(2.1)InstrInstanceDesc[set_required_begin_end_t] Input {0} has no lookback".format(getMetaId(input)))
                # elif input is None:
                #     print("(2.2)InstrInstanceDesc[set_required_begin_end_t] input is none")
                # else:
                #     print("(2.2)InstrInstanceDesc[set_required_begin_end_t] input.input_list is none.")
        if max_lookback:
            cum_lookback.set_input_lookback(max_lookback)

        # update the min/max of input date.
        if cum_lookback.input_begin_t < self.input_begin_t:
            self.input_begin_t = cum_lookback.input_begin_t

        if cum_lookback.input_end_t > self.input_end_t:
            self.input_end_t = cum_lookback.input_end_t

        # print("(3)InstrInstanceDesc[set_required_begin_end_t] Input setted:[{0}:{1}] for{2}".format(str(self.input_begin_t), str(self.input_end_t), getMetaId(self)))

        self.cum_lookback_cache = SliceCache(required_begin_t, required_end_t, cum_lookback)
        return cum_lookback


    def set_required_begin_end_t_and_ret_protobuf(self, required_begin_gs_int, required_end_gs_int):
        # print(
        #     "Required[{0}:{1}], available[{2}:{3} for:{4}".format(str(required_begin_gs_int), str(required_end_gs_int),
        #                                                           str(self.min_t), str(self.max_t), getMetaId(self)))
        if self.min_t is not None:
            if required_begin_gs_int < self.min_t:
                required_begin_gs_int = self.min_t
            elif required_begin_gs_int > self.max_t:
                required_begin_gs_int = self.max_t

        if self.max_t is not None:
            if required_end_gs_int > self.max_t:
                required_end_gs_int = self.max_t
            elif required_end_gs_int < self.min_t:
                required_end_gs_int = self.min_t

        if required_end_gs_int < required_begin_gs_int:
            required_end_gs_int = required_begin_gs_int  # just reset the end t to end t

        # print(
        #         "at last, required[{0}:{1}], available[{2}:{3} for:{4}".format(str(required_begin_gs_int),
        #                                                               str(required_end_gs_int),
        #                                                               str(self.min_t), str(self.max_t),
        #                                                               getMetaId(self)))
            # raise Exception(
            #     "No data available, required:[" + str(required_begin_t) + "-" + str(required_end_t) + "] data:[" + str(self.min_t) + "-" + str(self.max_t) + "].")
        cum_lookback = self.set_required_begin_end_t(gftIO.gs_day_2_pd_dt(required_begin_gs_int),
                                                     gftIO.gs_day_2_pd_dt(required_end_gs_int))

        resultMeta = instStream.InstResultMeta()
        resultMeta.lookback = math.ceil((cum_lookback.input_end_t - cum_lookback.input_begin_t).total_seconds() / 86400)
        if self.max_t:
            resultMeta.max_t = self.max_t
            if self.max_j_gid:
                resultMeta.max_t_j = self.max_j_gid

        if self.min_t:
            resultMeta.min_t = self.min_t
            if self.min_j_gid:
                resultMeta.min_t_j = self.min_j_gid

        resultMeta.calc_min_t = required_begin_gs_int
        resultMeta.calc_max_t = required_end_gs_int
        cum_lookback.add_self_to_result_meta(resultMeta)
        return resultMeta.SerializeToString()

        # return required_begin_gs_int, required_end_gs_int, get_gs_time_value(cum_lookback.input_begin_t), get_gs_time_value(cum_lookback.input_end_t)

    # if return false, will call the function: slice_my_output below.
    def is_out_put_sliced_to_required_date(self):
        return True

    def get_input_begin_end_t(self):
        return get_gs_time_value(self.input_begin_t), get_gs_time_value(self.input_end_t), self.get_pt_name()

    def get_all_t_and_pt_col_name(self):
        return get_gs_time_value(self.input_begin_t), get_gs_time_value(self.input_end_t), get_gs_time_value(
            self.required_begin_t), get_gs_time_value(self.required_end_t), self.get_pt_name()

    def get_input_begin_as_gs_int(self):
        return get_gs_time_value(self.input_begin_t)

    def get_pt_name(self):
        if self.primary_t is not None:
            return self.primary_t.col_name
        return None

    def get_data_min_max_t_and_pt_name(self):
        return get_gs_time_value(self.required_begin_t), get_gs_time_value(self.required_end_t), self.get_pt_name()

    @classmethod
    def create_result_meta(cls, j_gid, input_lookback, primary_t, col_metas, min_t, max_t, min_j_gid, max_j_gid,
                           input_list,
                           inst_id=None):
        if input_list is None:
            list_size = -1
        else:
            list_size = len(input_list)
        # print("InstrInstanceDesc[create_result_meta], Create meta[{0}:{1}], input_list size:{3} for{2}".format(str(min_t), str(max_t), str(inst_id), str(list_size)))
        ret = cls()
        ret.set_data(j_gid, input_lookback, primary_t, col_metas, min_t, max_t, min_j_gid, max_j_gid, input_list,
                     inst_id)
        # print("Look back is :" + str(ret.input_lookback) + " for:" + getMetaId(ret))
        return ret

    @classmethod
    def create_dict_meta(cls, j_gid, min_t, max_t, min_j_gid, max_j_gid, last_kv_map, input_list, inst_id=None):
        ret = cls()
        ret.set_data(j_gid, invalid_lookback_step, None, None, min_t, max_t, min_j_gid, max_j_gid, input_list, inst_id)
        ret.kv_map = last_kv_map
        return ret

    # used for creating meta for scalar.
    @classmethod
    def create_scalar(cls, scalar, inst_id=None):
        ret = cls()
        ret.set_data(None, invalid_lookback_step, None, None, None, None, None, None, None, inst_id)
        ret.scalar = scalar
        return ret

    @classmethod
    def create_place_holder_idx(cls, place_holder, inst_id = None):
        ret = cls()
        ret.set_data(None, invalid_lookback_step, None, None, None, None, None, None, None, inst_id)
        ret.scalar = 0
        ret.place_holder = place_holder
        return ret

    @classmethod
    def create_one_meta_from_pb_def(cls, gid, pb_one_table: mtproto.tableMeta):
        # j_gid, primary_t, col_metas, min_t, max_t, min_j_gid, max_j_gid
        cols = list()
        for col in pb_one_table.cols:
            one_col = ColDesc.create_from_pb_col_meta(col)
            cols.append(one_col)

        if pb_one_table.HasField("primaryT"):
            primary_t = find_col_with_name(pb_one_table.primaryT, cols)
        else:
            # find if there is some T exists.
            primary_t = None
            # for col in cols:
            #     if col.col_type == 7: # PARAMETER_TYPE_DATETIME
            #         primary_t = col

        if pb_one_table.HasField("max_t"):
            max_t = pb_one_table.max_t
        else:
            max_t = None

        if pb_one_table.HasField("min_t"):
            min_t = pb_one_table.min_t
        else:
            min_t = None

        if pb_one_table.HasField("ptCalendar"):
            pb_calendar = pb_one_table.ptCalendar
            if pb_calendar.HasField('jGid'):
                j_gid = pb_calendar.jGid
            else:
                j_gid = gid
            if pb_calendar.HasField('tColumnName'):
                t_column_name = pb_calendar.tColumnName
            else:
                t_column_name = pb_one_table.primaryT

            if pb_calendar.HasField('regularDaysGid'):
                trade_calendar = Calendar(pb_calendar.regularDaysGid, None, j_gid, t_column_name)
            else:
                if pb_calendar.HasField('indexPickle'):
                    index = pickle.loads(pb_calendar.indexPickle)
                    trade_calendar = Calendar(None, index, j_gid, t_column_name)
                else:
                    raise Exception("None index found for gid:[{0}]", gid)
        else:
            trade_calendar = None
        lbs = LookbackStep(trade_calendar, 0, False)
        if 0 == len(cols):
            raise Exception("j:{0} has no cols, maybe the 'js_meta' on node is not configed!".format(gid))
        return InstrInstanceDesc.create_result_meta(gid, lbs, primary_t, cols, min_t, max_t, gid, gid, None)

    @classmethod
    def parse_js_from_pb_and_return_states(cls, pb_buff):
        pack = mtproto.jMetaPack()
        pack.ParseFromString(pb_buff)
        ret = dict()
        has_meta_dic = dict()
        for oneTable in pack.allMetas:
            meta = cls.create_one_meta_from_pb_def(oneTable.jGid, oneTable.meta)
            ret[oneTable.jGid] = meta
            has_meta_dic[oneTable.jGid] = (meta.trade_calendar and meta.trade_calendar.index is not None)
        return ret, has_meta_dic
        # each meta is an table
        # j_gid, primary_t, col_metas, min_t, max_t, min_j_gid, max_j_gid

    # call this before saving it in cache.
    # so after loading the cache has no look back requires
    def clean_input_lookback_4_cache(self):
        self.input_lookback.lookback_step = 0

    def input_lookback(self):
        return self.input_lookback

    def get_scalar_as_float(self, index):
        if isinstance(self.scalar, numbers.Number):
            return float(self.scalar)
        if isinstance(self.scalar, str):
            if self.scalar.isdigit():
                return float(self.scalar)
        raise Exception("parameter[" + index + "] is not an number")

    def get_scalar_as_int(self, index):
        if isinstance(self.scalar, numbers.Number):
            return int(self.scalar)
        if isinstance(self.scalar, str):
            if self.scalar.isdigit():
                return int(self.scalar)
        raise Exception("parameter[" + index + "] is not an int")

    # clean the input list, so when saving it, it would save the would tree.
    def set_required_begin_end_and_retrieve_inputs(self, req_begin, req_end, input_set: set):
        if req_begin >= self.min_t:
            self.req_begin_time = req_begin
        else:
            self.req_begin_time = self.min_t

        if req_end <= self.max_t:
            self.req_end_time = req_end
        else:
            self.req_end_time = self.max_t

        if self.input_list:
            for input in self.input_list:
                if input:
                    input_set.add(input)

    def get_all_outpu_data(self):
        return self.input_lookback, self.min_t, self.max_t, self.min_j_gid, self.max_j_gid, self.min_t, self.max_t

    def as_dict(self):
        ret = self.__dict__.copy()
        if isinstance(self.primary_t, ColDesc):
            ret['primary_t'] = self.primary_t.as_dict()
        all_col = list()
        if self.col_metas:
            for col in self.col_metas:
                if isinstance(col, ColDesc):
                    all_col.append(col.as_dict())
                else:
                    all_col.append(col)
            ret['col_metas'] = all_col

        all_sub_meta = list()
        if self.input_list:
            for sub_meta in self.input_list:
                if isinstance(sub_meta, InstrInstanceDesc):
                    all_sub_meta.append(sub_meta.as_dict())
                else:
                    all_sub_meta.append(sub_meta)
            ret['input_list'] = all_sub_meta
        return ret

    def get_primary_t_name(self):
        if self.primary_t:
            if isinstance(self.primary_t, ColDesc):
                return self.primary_t.col_name
        return None

    def __getstate__(self):
        attr_dict = self.__dict__.copy()
        if self.input_list is not None:
            attr_dict['input_list'] = list()

        if attr_dict.__contains__('inst_id'):
            attr_dict['inst_id'] = 'Cache'

        attr_dict.pop('cum_lookback_cache', None)
        attr_dict.pop('trade_calendar', None)

        if self.primary_t is not None:
            attr_dict['primary_t'] = self.get_primary_t_name()
        return attr_dict

    def __setstate__(self, state):
        # print(state)
        self.__dict__.update(state)
        self.reset_primary_4_pickle()
        self.cum_lookback_cache = None
        if self.input_lookback:
            self.trade_calendar = self.input_lookback.cal
        else:
            self.trade_calendar = None
            # print("load is called")

    def __str__(self):
        return str(self.__dict__)

    def serialize_4_cpp(self):
        ret = pickle.dumps(self, -1)
        return ret

    def get_required_begin_time(self):
        return get_gs_time_value(self.required_begin_t)

    def reset_primary_4_pickle(self):
        if self.primary_t is not None:
            pt_name = self.primary_t
            self.primary_t = None
            for col in self.col_metas:
                if col.col_name == pt_name:
                    self.primary_t = col
                    return
            if self.primary_t is None:
                raise Exception("Can not found primary_t:" + pt_name + " in col metas")

    def copy_and_merge_old_data(self, old_desc):
        # print("New data:{0}, old data{1}".format(str(self.required_begin_t), str(old_desc.required_begin_t)))

        ret = copy.copy(self)
        if self.required_begin_t <= old_desc.required_begin_t:
            return ret
        else:
            ret.required_begin_t = old_desc.required_begin_t
        return ret
        # only move required_begin_t to the old data.
        # all the other data use the


def patch_calendar_and_ret_2_proto(data, meta: InstrInstanceDesc):
    if (meta.trade_calendar) and (meta.trade_calendar.regular_days_gid == irregular_freq_name):
        # print("Create calendar:{0},pt name:{1} ".format(meta.j_gid, str(meta.get_pt_name())))
        meta.trade_calendar.index = create_index(data, meta.get_pt_name())
    else:
        # print("Create calendar:{0},pt name:{1} ".format(meta.j_gid, str(meta.get_pt_name())))
        cal = create_calendar(data, meta.get_pt_name(), meta.j_gid)
        if cal is None:
            return None
        meta.trade_calendar = cal
        meta.input_lookback.cal = cal  #
    # print("Patch cal for j:{0},data:{1}".format(str(meta.j_gid), str(cal.index)))
    return cal.serialize_to_protobuf(mtproto.calendar(), True).SerializeToString()


class SubContext:
    # required_begin_date is from the request_msg
    def __init__(self, required_begin_date, required_end_date):
        self.required_begin_date = required_begin_date
        self.required_end_date = required_end_date

    def reset_input(self, my_meta: InstrInstanceDesc):
        self.input_datas = list()
        self.input_metas = list()
        self.my_meta = my_meta

    def add_input(self, data, meta: InstrInstanceDesc):
        self.input_datas.append(data)
        self.input_metas.append(meta)


class DataFactory:
    def __init__(self):
        self.all_raw_data = dict()
        # gid->begin->end
        self.data_with_time_map = dict()

    # data would saved in
    def get_data(self, gid, begin, end):
        return self.all_data.get(gid)


def copy_and_slice(input_data, begin_date, end_date):
    return input_data


def get_first_calendar(input_list):
    for input in input_list:
        if (input is not None) and (input.trade_calendar is not None):
            return input.trade_calendar
    return None


# get max of all min_t and min max_t
def get_max_min_t_and_min_max_t(input_list):
    max_min_t = 10000  # 10000 is the min date defined in c++ struct Timestamp
    min_j_gid = None

    min_max_t = 100000  # 1000000 is the max date defined in c++ struct Timestamp
    max_j_gid = None
    for meta in input_list:
        if meta is None:
            continue

        if meta.min_t and meta.min_t > max_min_t:
            max_min_t = meta.min_t
            min_j_gid = meta.min_j_gid

        if meta.max_t and meta.max_t < min_max_t:
            min_max_t = meta.max_t
            max_j_gid = meta.max_j_gid

    return max_min_t, min_max_t, min_j_gid, max_j_gid


def get_first_with_primary_t_and_cols(input_list):
    for input in input_list:
        if input and input.has_primary_t_and_col_metas():
            return input
    return None

def get_meta_with_max_num_of_cols(input_list):
    col_size = -1
    ret = None
    idx = 0
    for input in input_list:
        if input and input.has_primary_t_and_col_metas():
            if len(input.col_metas) > col_size:
                ret = input
        # else:
        #     print("Input {0} has no pt or colmetas".format(str(idx)))
        idx += 1
    return ret



def get_gs_time_value(timestamp: pd.Timestamp):
    return int(timestamp.tz_localize(tz="UTC").timestamp() / 86400) + 62091


# represent the basic data requirement of one instruction.
# so each code would have it's own logic for it.
# by default ,the wavefront python framework would call get_instruction_meta() to get it.
# in each instruction, would have a code block like this.
# def get_instruction_meta():
#     return InstructionDefault()
# meta generator.
class DescGenerator:
    # all these method begin will be called in the first routine of wavefront calculation
    def __init__(self):
        self.b_need_append_lookback = False
        self.b_need_move_observer_t = False

    @classmethod
    def create_default(cls):
        ret = cls()
        return ret

    @classmethod
    def create_with_lookback_para(cls, lookback_para_idx):
        ret = cls()
        ret.set_lookback_para_idx(lookback_para_idx)
        return ret

    @classmethod
    def create_with_lookback_and_observer_para(cls, lookback_para_idx, move_observer_para_idx, move_forward):
        ret = cls()
        ret.b_need_append_lookback = True
        ret.i_lookback_period_para_idx = lookback_para_idx
        ret.b_need_move_observer_t = True
        ret.i_move_observer_para_idx = move_observer_para_idx
        ret.b_forward_move_observer_t = move_forward

    import os.path

    def get_instr_desc(self, gid, input_list, inst_id=None):
        input_lookback_step = self.get_look_back_step(input_list)
        # print("DescGenerator[get_instr_desc], Lookback is:{0} for inst:{1}".format(str(input_lookback_step), str(inst_id)))
        lookforward = self.get_look_forward_peroid(input_list)
        ret_meta = get_first_with_primary_t_and_cols(input_list)
        if ret_meta is None:
            primary_t = None
            col_metas = None
        else:
            primary_t = ret_meta.primary_t
            col_metas = ret_meta.col_metas
        min_t, max_t, min_j_gid, max_j_gid = get_max_min_t_and_min_max_t(input_list)

        if (input_lookback_step is not None) and input_lookback_step.lookback_step > 0:
            try:
                min_t_loc = input_lookback_step.cal.index.get_slice_bound(gftIO.gs_day_2_pd_dt(min_t), 'left',
                                                                          'loc') + input_lookback_step.lookback_step
            except KeyError:
                max_t_str = ''
                min_t_str = ''
                if input_lookback_step.cal is not None:
                    max_t_str = str(input_lookback_step.cal.index.max())
                    min_t_str = str(input_lookback_step.cal.index.min())
                # print(
                #     "input lookback:{0}, target:{3}, idx({4}):[{1}:{2}]".format(str(input_lookback_step.lookback_step),
                #                                                                 min_t_str, max_t_str,
                #                                                                 str(gftIO.gs_day_2_pd_dt(min_t)), str(
                #             input_lookback_step.cal.regular_days_gid)))
                file_name = input_lookback_step.cal.regular_days_gid + ".pkl"
                if not os.path.isfile(file_name):
                    data = dict()
                    data['cal'] = input_lookback_step
                    data['index'] = input_lookback_step.cal.index
                    data['inputs'] = input_list
                    gftIO.zdump(data, file_name)
                    # print("save input_lookback file:" + file_name)
                min_t_loc = input_lookback_step.cal.index.get_slice_bound(gftIO.gs_day_2_pd_dt(min_t), 'left',
                                                                          'loc') + input_lookback_step.lookback_step
            if min_t_loc < input_lookback_step.cal.index.size:
                min_t = get_gs_time_value(input_lookback_step.cal.index[min_t_loc])
            else:
                min_t = max_t + 1  # means there is not date available.
                if inst_id is None:
                    inst_id = "Unknown"
                if gid is None:
                    gid = "Unknonw"
                raise Exception("Not enough data for {0}[{1}]".format(str(inst_id), gid))

        if lookforward:
            max_t -= math.floor(lookforward)

        # print("Data available[{0}:{1}] for{2}".format(str(min_t),str(max_t),str(inst_id)))
        return InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                    min_j_gid,
                                                    max_j_gid, input_list, inst_id)

    def need_slice_data(self):
        return True

    def get_look_forward_peroid(self, input_list):
        return 0

    def get_use_prev_if_not_found(self):
        return False

    def get_look_back_step(self, input_list: list):
        calendar = get_first_calendar(input_list)
        if self.b_need_append_lookback:
            lookback = input_list[self.i_lookback_period_para_idx].get_scalar_as_int(self.i_lookback_period_para_idx)
        else:
            lookback = 0
        if calendar is not None:
            return LookbackStep(calendar, lookback, self.get_use_prev_if_not_found())
        # print("DescGenerator[get_look_back_step] return null")
        return None

    def need_move_observer_t(self):
        return False

    def move_observer_para_idx(self):
        return 0

    def forward_move_observer(self):
        return False

    def set_lookback_para_idx(self, para_idx):
        self.b_need_append_lookback = True
        self.i_lookback_period_para_idx = para_idx

    def set_move_observer(self, para_idx, move_forward):
        self.b_need_move_observer_t = False
        self.b_forward_move_observer_t = move_forward
        self.i_move_observer_para_idx = para_idx

    def get_meta_begin_time(self, input_list, required_begin_time, input_lookback):
        return required_begin_time - input_lookback

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
        my_required_begin = context.required_begin_date - context.my_meta.input_lookback
        for meta in context.input_metas:
            if meta.mint_t:
                if my_required_begin < meta.min_t:
                    my_required_begin = meta.min_t
        return my_required_begin + context.my_meta.input_lookback  # here move forward the begin data with input_lookback period

    # override this method if you wanna define your own logic for end_date
    def get_slice_end_date(self, context: SubContext):
        my_required_end = context.required_begin_date
        for meta in context.input_metas:
            if meta.max_t:
                if my_required_end > meta.max_t:
                    my_required_end = meta.max_t
        return my_required_end


class ResampleDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        data = input_list[3]
        freq = input_list[4]
        lookback_step = input_list[5].get_scalar_as_int(5)
        primary_t = freq.primary_t

        col_metas = list()
        for col in data.col_metas:
            if col.col_type == primary_t.col_type:
                col_metas.append(primary_t)
            else:
                col_metas.append(col)
        # begin is limited to input 0
        # end is expanded to input 1
        if data.min_t <= freq.min_t:
            min_t = freq.min_t
            min_j_gid = freq.j_gid
        else:
            min_t = gftIO.pd_dt_2_gs_day(
                freq.trade_calendar.index[freq.trade_calendar.index.get_slice_bound(gftIO.gs_day_2_pd_dt(data.min_t), 'left', 'loc')])
            # print("Freq min_t:" + str(min_t))
            min_j_gid = data.min_j_gid

        resample_end_date = data.max_t + lookback_step
        if resample_end_date >= freq.max_t:
            max_t = freq.max_t
            max_j_gid = freq.j_gid
        else:
            max_t = gftIO.pd_dt_2_gs_day(freq.trade_calendar.index[freq.trade_calendar.index.get_slice_bound(gftIO.gs_day_2_pd_dt(resample_end_date), 'right', 'loc')-1])
            # print("Freq max_t:" + str(max_t))
            max_j_gid = data.j_gid

        # print("Time:[{0},{1}]".format(str(min_t),str(max_t)))

        input_lookback_step = copy.copy(input_list[4].input_lookback)
        input_lookback_step.lookback_step = lookback_step

        # print("Resample input_lookback is:" + str(input_lookback_step))
        ret = InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                   min_j_gid,
                                                   max_j_gid, input_list, inst_id)
        return ret


class Transform2ScalarDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        ret = InstrInstanceDesc.create_scalar(0, inst_id)
        ret.input_list = input_list
        return ret

    def need_slice_data(self):
        return False


# treat as a scalar.
class DictDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        if input_list[2] is not None:
            # the 3rd input should be key value maps.
            kv_map = input_list[2].kv_map.copy()
        else:
            kv_map = dict()

        kv_map[input_list[0].scalar] = input_list[1]  # put the key value into kv_map

        min_t, max_t, min_j_gid, max_j_gid = get_max_min_t_and_min_max_t(input_list)

        # print("DictDescGenerator[get_instr_desc]")
        return InstrInstanceDesc.create_dict_meta(gid, min_t, max_t, min_j_gid, max_j_gid, kv_map, input_list, inst_id)

    def need_slice_data(self):
        return False


class ExtractDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def need_slice_data(self):
        return False

    def get_instr_desc(self, gid, input_list, inst_id=None):
        # print("Extract desc of:"+ input_list[1].scalar)
        if (input_list[0] is not None) and (input_list[0].kv_map):
            ret = input_list[0].kv_map.get(input_list[1].scalar, None)
            if ret is not None:
                ret.inst_id = inst_id
                return ret
        raise Exception("Can not find key:" + str(input_list[1]))

class RaiseDimension(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def need_slice_data(self):
        return False

    def get_instr_desc(self, gid, input_list, inst_id=None):
        # print("Raise dimension,add key as columns")
        if not isinstance(input_list[0], InstrInstanceDesc):
            raise Exception("Input[0] is not a dict")
        dic = input_list[0].kv_map
        if dic.__len__() != 1:
            raise Exception("Input[0] has more than one item")

        for key, val in dic.items():
            ret = copy.copy(val)
            ret.inst_id = inst_id
            ret.col_metas = copy.copy(val.col_metas)
            ret.input_list = input_list
            # 4 is O,
            ret.col_metas.append(ColDesc('ADDED',4, None, None,None, None,None, None,None, None))
            return ret



class LeadDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        self.set_lookback_para_idx(4)
        return


def get_cal_omit_first(input_list):
    not_first = False
    for input in input_list:
        if not_first:
            if (input is not None) and (input.trade_calendar is not None):
                return input.trade_calendar
        else:
            not_first = True
    # in the worst cases, return the frist one
    # print("Found no calendar in input.")
    return input_list[0].trade_calendar


class CppDescGenerator(DescGenerator):
    def get_look_back_step(self, input_list: list):
        calendar = get_cal_omit_first(input_list)
        if self.b_need_append_lookback:
            lookback = input_list[self.i_lookback_period_para_idx].get_scalar_as_int(self.i_lookback_period_para_idx)
        else:
            lookback = 0
        return LookbackStep(calendar, lookback, self.get_use_prev_if_not_found())

    @classmethod
    def create_with_lookback_para(cls, lookback_para_idx):
        ret = cls()
        ret.set_lookback_para_idx(lookback_para_idx)
        return ret

    def get_instr_desc(self, gid, input_list, inst_id=None):
        input_lookback_step = self.get_look_back_step(input_list)
        # print("DescGenerator[get_instr_desc], Lookback is:{0} for inst:{1}".format(str(input_lookback_step), str(inst_id)))
        lookforward = self.get_look_forward_peroid(input_list)
        ret_meta = get_meta_with_max_num_of_cols(input_list)
        if ret_meta is None:
            primary_t = None
            col_metas = None
        else:
            primary_t = ret_meta.primary_t
            col_metas = ret_meta.col_metas
        min_t, max_t, min_j_gid, max_j_gid = get_max_min_t_and_min_max_t(input_list)

        if (input_lookback_step is not None) and input_lookback_step.lookback_step > 0:
            try:
                min_t_loc = input_lookback_step.cal.index.get_slice_bound(gftIO.gs_day_2_pd_dt(min_t), 'left',
                                                                          'loc') + input_lookback_step.lookback_step
            except KeyError:
                max_t_str = ''
                min_t_str = ''
                if input_lookback_step.cal is not None:
                    max_t_str = str(input_lookback_step.cal.index.max())
                    min_t_str = str(input_lookback_step.cal.index.min())
                # print(
                #     "input lookback:{0}, target:{3}, idx({4}):[{1}:{2}]".format(str(input_lookback_step.lookback_step),
                #                                                                 min_t_str, max_t_str,
                #                                                                 str(gftIO.gs_day_2_pd_dt(min_t)), str(
                #             input_lookback_step.cal.regular_days_gid)))
                file_name = input_lookback_step.cal.regular_days_gid + ".pkl"
                if not os.path.isfile(file_name):
                    data = dict()
                    data['cal'] = input_lookback_step
                    data['index'] = input_lookback_step.cal.index
                    data['inputs'] = input_list
                    gftIO.zdump(data, file_name)
                    # print("save input_lookback file:" + file_name)
                min_t_loc = input_lookback_step.cal.index.get_slice_bound(gftIO.gs_day_2_pd_dt(min_t), 'left',
                                                                          'loc') + input_lookback_step.lookback_step
            if min_t_loc < input_lookback_step.cal.index.size:
                min_t = get_gs_time_value(input_lookback_step.cal.index[min_t_loc])
            else:
                min_t = max_t + 1  # means there is not date available.
                if inst_id is None:
                    inst_id = "Unknown"
                if gid is None:
                    gid = "Unknonw"
                raise Exception("Not enough data for {0}[{1}]".format(str(inst_id), gid))

        if lookforward:
            max_t -= math.floor(lookforward)

        # print("Data available[{0}:{1}] for{2}".format(str(min_t),str(max_t),str(inst_id)))
        return InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                    min_j_gid,
                                                    max_j_gid, input_list, inst_id)




def load_meta(filename):
    with open(filename, "rb") as fpz:
        data = fpz.read()
        return InstrInstanceDesc.parse_multi_js_from_protobuf(data)


class CppJoinDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        input_lookback_step = self.get_look_back_step(input_list)
        lookforward = self.get_look_forward_peroid(input_list)
        ret_meta = get_first_with_primary_t_and_cols(input_list)
        if ret_meta is None:
            primary_t = None
            col_metas = None
        else:
            primary_t = ret_meta.primary_t
            col_metas = ret_meta.col_metas
        min_t, max_t, min_j_gid, max_j_gid = get_max_min_t_and_min_max_t(input_list)

        if input_lookback_step and input_lookback_step.lookback_step > 0:
            min_t_loc = input_lookback_step.cal.index.get_slice_bound(gftIO.gs_day_2_pd_dt(min_t), 'left',
                                                                      'loc') + input_lookback_step.lookback_step
            if min_t_loc < input_lookback_step.cal.index.size:
                min_t = get_gs_time_value(input_lookback_step.cal.index[min_t_loc])
            else:
                min_t = max_t + 1  # means there is not date available.
                if inst_id is None:
                    inst_id = "Unknown"
                if gid is None:
                    gid = "Unknonw"
                raise Exception("Not enough data for {0}[{1}]".format(str(inst_id), gid))

        if lookforward:
            max_t -= math.floor(lookforward)

        # print("Data available[{0}:{1}] for{2}".format(str(min_t),str(max_t),str(inst_id)))
        return InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                    min_j_gid,
                                                    max_j_gid, input_list, inst_id)


class PythonResampleDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        data = input_list[0]
        freq = input_list[1]
        if freq is None:
            freq = input_list[0]
        primary_t = freq.primary_t

        col_metas = list()
        for col in data.col_metas:
            if col.col_type == primary_t.col_type:
                col_metas.append(primary_t)
            else:
                col_metas.append(col)
        # begin is limited to input 0
        # end is expanded to input 1
        if data.min_t < freq.min_t:
            min_t = freq.min_t
            min_j_gid = freq.j_gid
        else:
            min_t = data.min_t
            min_j_gid = data.min_j_gid

        max_t = freq.max_t
        max_j_gid = freq.j_gid

        input_lookback_step = copy.copy(freq.input_lookback)

        if input_list[2] is None:
            input_lookback_step.lookback_step = 9999
        else:
            input_lookback_step.lookback_step = input_list[2].get_scalar_as_int(2)
            if input_lookback_step.lookback_step < 0:
                input_lookback_step.lookback_step = 9999

        # print("Resample input_lookback is:" + str(inst_id) + " gid:" + gid)
        ret = InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                   min_j_gid,
                                                   max_j_gid, input_list, inst_id)
        return ret


class OOTVResampleFirstColumnDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return

    def get_instr_desc(self, gid, input_list, inst_id=None):
        data = input_list[0]
        primary_t = data.primary_t

        col_metas = list()
        for col in data.col_metas:
            if col.col_type == primary_t.col_type:
                col_metas.append(primary_t)
            else:
                col_metas.append(col)
        # begin is limited to input 0
        # end is expanded to input 1

        min_t = data.min_t
        min_j_gid = data.min_j_gid

        max_t = data.max_t
        max_j_gid = data.j_gid

        daily_name = 'DAILY'
        daily_cal = Calendar(daily_name, get_freq(daily_name).index, daily_name, daily_name)
        input_lookback_step = LookbackStep(daily_cal, 9999, False)

        # print("Resample input_lookback is:" + str(input_lookback_step))
        ret = InstrInstanceDesc.create_result_meta(gid, input_lookback_step, primary_t, col_metas, min_t, max_t,
                                                   min_j_gid,
                                                   max_j_gid, input_list, inst_id)
        return ret



class AddSetDescGenerator(DescGenerator):
    def __init__(self):
        DescGenerator.__init__(self)
        return
    def get_instr_desc(self, gid, input_list, inst_id=None):
        old_dict = input_list[0]
        if old_dict is not None:
            old_kv_map = old_dict.kv_map
        else:
            old_kv_map = dict()

        set_name = input_list[1].scalar

        new_set = dict()
        for i in range(2,len(input_list)):
            data = input_list[i]
            new_set[data.j_gid] = data

        old_kv_map[set_name] = new_set

        get_max_min_t_and_min_max_t(input_list)
        min_t, max_t, min_j_gid, max_j_gid = get_max_min_t_and_min_max_t(input_list)
        # print("Resample input_lookback is:" + str(input_lookback_step))
        ret = InstrInstanceDesc.create_dict_meta(gid,min_t, max_t,min_j_gid, max_j_gid, old_kv_map, input_list, inst_id)
        return ret

        def need_slice_data(self):
            return False


def create_desc_generator_4_cpp():
    empty_code_meta = DescGenerator.create_default()
    pts_2nd_index = CppDescGenerator.create_with_lookback_para(4)  # since the first 3 parameter is used for O/T/num
    resample_code_meta = ResampleDescGenerator()
    transform_2_scalar_code_meta = Transform2ScalarDescGenerator()
    lead_code_meta = CppDescGenerator.create_with_lookback_para(4)
    dict_meta = DictDescGenerator()
    extract_meta = ExtractDescGenerator()
    cpp_with_global_t = CppDescGenerator()
    raise_dimension_gen = RaiseDimension()
    add_set_gen = AddSetDescGenerator()
    return empty_code_meta, pts_2nd_index, resample_code_meta, transform_2_scalar_code_meta, lead_code_meta, dict_meta, extract_meta, cpp_with_global_t, raise_dimension_gen, add_set_gen
