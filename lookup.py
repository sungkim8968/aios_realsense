import aios
import time
import threading
import numpy as np
import json

Server_IP_list = []


def main():

    Server_IP_list = aios.broadcast_func()
    # print(Server_IP_list[0])
    # if Server_IP_list:

    for i in range(len(Server_IP_list)):
        # print(Server_IP_list[i])
        aios.getCVP_pt("10.10.20.93")

    # aios.getRoot(Server_IP_list[i])
        # aios.getMotionCtrlConfig(Server_IP_list[i], 1)
        # aios.getMotorConfig(Server_IP_list[i], 1)
    # aios.set


if __name__ == '__main__':
    main()
