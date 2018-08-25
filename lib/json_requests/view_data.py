from lib.gftTools import gftIO


def create_view_data_req(target_gid, begin_date, end_date, file_name):
    begin_date = gftIO.pd_dt_2_gs_day(begin_date)
    end_date = gftIO.pd_dt_2_gs_day(end_date)
    req_str = '''{
	"all_j": [{
		"data_j": {
			"nid": 0,
			"props": {
				"_gid": "%s"
			}
		},
		"filter": [{
			"filter_type": 1,
			"para_t": [%d,
			%d]
		}],
		"result_idx": 1
	}],
	"para": {
		"begin_date": %d,
		"end_date": %d,
		"i_node_result_idx": 0
	},
	"perfCheck": 0,
	"useCache": 0,
	"write2File":%s
}'''
    return req_str % (target_gid, begin_date, begin_date+2, begin_date, end_date, file_name)


import pandas as pd
import os.path

data_root_path = '/home/gft/work/data/'

def get_data_from_gs(gscall, jgid, start_date, end_date, filename=None):
    if filename is None or len(filename):
        filename = jgid + '.pkl'
        
    if isinstance(start_date, str):
        start_ts = pd.Timestamp(start_date)
    elif not isinstance(start_date, pd.Timestamp):
        raise Exception("Start date must be pd.Timestamp or str")
    else:
        start_ts = start_date

    if isinstance(end_date, str):
        end_ts = pd.Timestamp(end_date)
    elif not isinstance(end_date, pd.Timestamp):
        raise Exception("End date must be pd.Timestamp or str")
    else:
        end_ts = start_date

    sever_ret = gscall.call_vq(4, 5, create_view_data_req(jgid, start_ts, end_ts, filename))
    filepath = data_root_path+filename
    if os.path.exists(filepath):
        return gftIO.zload(filepath)
    raise Exception("Load j from gs failed, server ret is : "+ str(sever_ret))