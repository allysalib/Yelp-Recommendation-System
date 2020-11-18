#!/usr/bin/env python

import pandas as pd
pd.set_option('display.max_columns', None)
import os, json
import glob
import numba
import concurrent.futures
from lenskit.datasets import MovieLens
from lenskit.crossfold import partition_users, SampleFrac
from lenskit.algorithms.basic import Bias
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms import Recommender
from lenskit.metrics.predict import rmse, mae
from lenskit.metrics.topn import recip_rank, precision, recall, ndcg
from lenskit.batch import predict, recommend
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn

json_file_list_subset = ['/scratch/as12453/all_data_restaurants_subset.json/part-00000-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00001-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00002-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00003-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00004-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00005-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00006-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00007-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00008-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00009-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00010-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00011-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00012-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00013-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00014-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00015-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00016-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00017-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00018-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00019-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00020-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00021-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00022-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00023-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00024-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00025-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00026-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00027-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00028-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00029-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00030-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00031-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00032-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00033-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00034-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00035-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00036-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00037-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00038-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00039-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00040-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00041-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00042-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00043-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00044-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00045-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00046-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00047-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00048-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00049-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00050-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00051-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00052-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00053-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00054-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00055-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00056-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00057-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00058-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00059-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00060-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00061-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00062-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00063-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00064-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00065-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00066-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00067-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00068-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00069-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00070-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00071-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00072-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00073-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00074-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00075-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00076-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00077-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00078-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00079-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00080-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00081-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00082-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00083-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00084-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00085-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00086-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00087-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00088-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00089-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00090-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00091-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00092-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00093-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00094-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00095-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00096-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00097-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00098-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00099-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00100-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00101-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00102-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00103-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00104-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00105-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00106-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00107-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00108-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00109-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00110-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00111-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00112-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00113-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00114-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00115-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00116-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00117-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00118-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00119-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00120-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00121-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00122-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00123-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00124-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00125-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00126-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00127-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00128-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00129-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00130-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00131-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00132-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00133-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00134-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00135-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00136-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00137-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00138-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00139-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00140-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00141-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00142-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00143-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00144-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00145-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00146-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00147-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00148-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00149-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00150-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00151-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00152-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00153-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00154-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00155-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00156-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00157-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00158-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00159-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00160-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00161-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00162-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00163-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00164-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00165-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00166-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00167-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00168-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00169-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00170-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00171-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00172-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00173-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00174-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00175-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00176-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00177-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00178-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00179-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00180-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00181-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00182-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00183-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00184-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00185-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00186-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00187-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00188-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00189-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00190-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00191-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00192-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00193-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00194-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00195-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00196-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00197-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00198-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json',
'/scratch/as12453/all_data_restaurants_subset.json/part-00199-939eb3a6-ced3-4f17-ae7d-20ac2a0974c1-c000.json']

dfs = []

for file in json_file_list_subset:
    data = pd.read_json(file, lines=True, orient = 'columns')
    dfs.append(data)
    
full_data = pd.concat(dfs, ignore_index=True)

ratings_data = full_data[['user_id', 'business_id', 'review_stars']].rename(columns={'user_id':'user', 'business_id':'item', 'review_stars':'rating'})
ratings_data['rating_binary'] = 0
ratings_data.loc[(ratings_data['rating'] > 3), 'rating_binary'] = 1

def fit_eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    model = fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs_10 = batch.recommend(model, users, 10)
    recs_100 = batch.recommend(model, users, 100)
    # add the algorithm name for analyzability
    recs_10['Algorithm'] = aname
    recs_100['Algorithm'] = aname
    return model, recs_10, recs_100

def test_eval(model, test):
    users = test.user.unique()
    recs_10 = batch.recommend(model, users, 10)
    recs_100 = batch.recommend(model, users, 100)
    return recs_10, recs_100


def main():
    damping_values = [0, 1, 2, 5, 10]

    bias_test_data = []
    bias_validation_prediction_scores_list = []
    bias_validation_evals_list = []
    bias_models = []

    for d in damping_values:
        for train_val, test in partition_users(ratings_data, 1, SampleFrac(0.2, rng_spec=13), rng_spec=13):
            bias_test_data.append(test)
            for train, val in partition_users(train_val, 1, SampleFrac(0.2, rng_spec=13), rng_spec=13):
                B = Bias(items=True, users=False, damping=d)
                model, recs_10, recs_100 = fit_eval("Bias, Damping={}".format(d), B, train, val)
                bias_models.append([d, model])
                predictions = model.predict(val[['user', 'item']])
                rmse_score = rmse(predictions, val['rating'])
                mae_score = mae(predictions, val['rating'])
                bias_validation_prediction_scores_list.append([d, rmse_score, mae_score])
                val_binary = val[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
                rla = topn.RecListAnalysis()
                rla.add_metric(topn.recip_rank)
                rla.add_metric(topn.precision)
                rla.add_metric(topn.recall)
                rla.add_metric(topn.ndcg)
                results_10recs = rla.compute(recs_10, val_binary)
                evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
                results_100recs = rla.compute(recs_100, val_binary)
                evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
                evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
                bias_validation_evals_list.append([d, evals_full])

    bias_validation_prediction_scores = pd.DataFrame(bias_validation_prediction_scores_list, columns = ['damping_factor', 'rmse', 'mae'])              

    bias_validation_evals = []
    
    for i in range(len(bias_validation_evals_list)):
        data = bias_validation_evals_list[i][1]
        data['damping_factor'] = bias_validation_evals_list[i][0]
        bias_validation_evals.append(data)

    bias_validation_evals_df = pd.concat(bias_validation_evals, ignore_index=True)

    print("Bias models validation prediction scores by damping factor:")
    print(bias_validation_prediction_scores)

    bias_validation_evals_aggregated = bias_validation_evals_df.groupby(['damping_factor']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Bias models validation evaluations by damping factor:")
    print(bias_validation_evals_aggregated)

    #chose parameter to opt:
    best_bias_params = bias_validation_evals_aggregated['ndcg']['mean'].idxmax()
    print("Best Hyperparameters:")
    print("Damping value = {}".format(best_bias_params))

    best_bias_models = []

    for b in bias_models:
        if b[0] == best_bias_params: best_bias_models.append(b)

    bias_test_prediction_scores_list = []
    bias_test_evals_list = []

    print("Best Bias Model Test Evaluations:")

    for i in best_bias_models:
        model = i[1]
        test_data = bias_test_data[0]
        predictions = model.predict(test_data[['user', 'item']])
        rmse_score = rmse(predictions, test_data['rating'])
        mae_score = mae(predictions, test_data['rating'])
        bias_test_prediction_scores_list.append([rmse_score, mae_score])
        recs_10, recs_100 = test_eval(model, test_data)
        test_binary = test_data[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank)
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.ndcg)
        results_10recs = rla.compute(recs_10, test_binary)
        evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
        results_100recs = rla.compute(recs_100, test_binary)
        evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
        evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
        bias_test_evals_list.append(evals_full)

    bias_test_prediction_scores = pd.DataFrame(bias_test_prediction_scores_list, columns = ['rmse', 'mae'])              

    bias_test_evals = bias_test_evals_list[0]

    print("Best Bias model test aggregated prediction scores:")
    print(bias_test_prediction_scores)

    bias_test_evals_agg = bias_test_evals.agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Best Bias model test aggregated evaluations:")
    print(bias_test_evals_agg)

    regularization_values = [0, 0.1, 0.5, 1]
    dimensionality_values = [50, 250, 500]

    biasMF_test_splits = []
    biasMF_validation_prediction_scores_list = []
    biasMF_validation_evals_list = []
    biasMF_models = []

    for r in regularization_values:
        for f in dimensionality_values:
            count = 0
            for train_val, test in partition_users(ratings_data, N_SPLIT, SampleFrac(FRAC_SPLIT, rng_spec=13), rng_spec=13):
                biasMF_test_splits.append(test)
                for train, val in partition_users(train_val, 1, SampleFrac(FRAC_SPLIT, rng_spec=13), rng_spec=13):
                    count += 1
                    BMF = BiasedMF(features=f, reg=r, rng_spec=13)
                    model, recs_10, recs_100 = fit_eval("BiasMF, Reg={},Dim={}".format(r,f), BMF, train, val)
                    biasMF_models.append([(count-1), r, f, model])
                    predictions = model.predict(val[['user', 'item']])
                    rmse_score = rmse(predictions, val['rating'])
                    mae_score = mae(predictions, val['rating'])
                    biasMF_validation_prediction_scores_list.append([(count-1), r, f, rmse_score, mae_score])
                    val_binary = val[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
                    rla = topn.RecListAnalysis()
                    rla.add_metric(topn.recip_rank)
                    rla.add_metric(topn.precision)
                    rla.add_metric(topn.recall)
                    rla.add_metric(topn.ndcg)
                    results_10recs = rla.compute(recs_10, val_binary)
                    evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
                    results_100recs = rla.compute(recs_100, val_binary)
                    evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
                    evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
                    biasMF_validation_evals_list.append([(count-1), r, f, evals_full])

    biasMF_validation_prediction_scores = pd.DataFrame(biasMF_validation_prediction_scores_list, columns = ['split', 'regularization', 'features', 'rmse', 'mae'])              

    biasMF_validation_evals = []

    for i in range(len(biasMF_validation_evals_list)):
        data = biasMF_validation_evals_list[i][3]
        data['regularization'] = biasMF_validation_evals_list[i][1]
        data['features'] = biasMF_validation_evals_list[i][2]
        data['split'] = biasMF_validation_evals_list[i][0]
        biasMF_validation_evals.append(data)

    biasMF_validation_evals_df = pd.concat(biasMF_validation_evals, ignore_index=True)

    print("BiasMF models validation prediction scores by split:")
    print(biasMF_validation_prediction_scores)

    biasMF_validation_evals_splits = biasMF_validation_evals_df.groupby(['regularization', 'features', 'split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("BiasMF models validation evaluations by split:")
    print(biasMF_validation_evals_splits)

    biasMF_validation_prediction_scores_aggregated = biasMF_validation_prediction_scores.groupby(['regularization', 'features']).agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated BiasMF models validation prediction scores by parameters:")
    print(biasMF_validation_prediction_scores_aggregated)

    biasMF_validation_evals_aggregated = biasMF_validation_evals_df.groupby(['regularization', 'features']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Aggregated BiasMF models validation evaluations by parameters:")
    print(biasMF_validation_evals_aggregated)

    #chose parameter to opt:
    best_biasMF_params = biasMF_validation_evals_aggregated['ndcg']['mean'].idxmax()
    print(best_biasMF_params)
    print("Best Hyperparameters:")
    print("Regularization = {}, Features = {}".format(best_biasMF_params[0], best_biasMF_params[1]))

    best_biasMF_models = []

    for b in biasMF_models:
        if (b[1] == best_biasMF_params[0] and b[2] == best_biasMF_params[1]) : best_biasMF_models.append(b)

    biasMF_test_prediction_scores_list = []
    biasMF_test_evals_list = []

    print("Best BiasMF Model Test Evaluations:")

    for i in best_biasMF_models:
        model = i[3]
        count = i[0]
        test_data = biasMF_test_splits[count]
        predictions = model.predict(test_data[['user', 'item']])
        rmse_score = rmse(predictions, test_data['rating'])
        mae_score = mae(predictions, test_data['rating'])
        biasMF_test_prediction_scores_list.append([count, rmse_score, mae_score])
        test_binary = test_data[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
        recs_10, recs_100 = test_eval(model, test_data)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank)
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.ndcg)
        results_10recs = rla.compute(recs_10, test_binary)
        evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
        results_100recs = rla.compute(recs_100, test_binary)
        evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
        evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
        biasMF_test_evals_list.append([count, evals_full])

    biasMF_test_prediction_scores = pd.DataFrame(biasMF_test_prediction_scores_list, columns = ['split', 'rmse', 'mae'])              

    biasMF_test_evals = []

    for i in range(len(biasMF_test_evals_list)):
        data = biasMF_test_evals_list[i][1]
        data['split'] = biasMF_test_evals_list[i][0]
        biasMF_test_evals.append(data)

    biasMF_test_evals_df = pd.concat(biasMF_test_evals, ignore_index=True)

    print("BiasMF models test prediction scores by split:")
    print(biasMF_test_prediction_scores)

    biasMF_test_evals_splits = biasMF_test_evals_df.groupby(['split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("BiasMF models test evaluations by split:")
    print(biasMF_test_evals_splits)

    biasMF_test_prediction_scores_aggregated = biasMF_test_prediction_scores.agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated BiasMF models test prediction scores for best parameters:")
    print(biasMF_test_prediction_scores_aggregated)

    biasMF_test_evals_aggregated = biasMF_test_evals_df.agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Aggregated BiasMF models test evaluations for best parameters:")
    print(biasMF_test_evals_aggregated)
    
if __name__ == '__main__':
    main()
