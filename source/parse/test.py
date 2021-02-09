import parse.func as func
import numpy as np
import matplotlib.pyplot as plt

testegg=True
testtestbed=False
if testtestbed:
    path_to_data='/Users/ziegler/p8/psu_data/CAEN_wavedumps/8212020_1.0_2/'
    antennakey={0:[0,20,40,60,85,105,155,175],1:[0,30,50,70,90,115,135,170],
    2:[0,30,50,70,100,125,145,165]}

    parse_data=func.parse(path_to_data,antennakey)
    combined_data,phis=func.combine_and_calc_phis(parse_data)
    array_data=func.generate_array_data(combined_data)

    print(np.sort(np.array(list(array_data[0].keys()))))
if testegg:
    egg_path='/Users/ziegler/p8/locust_files/locust_mc_Seed370_LO25.8781G_Radius0.100_Pos0.000.egg'
    data=func.parse_egg(egg_path)

    slice_num=3
    slice_size=8194

    sliced_data=func.slice_egg(data,slice_num,slice_size)

    print(sliced_data.shape)
