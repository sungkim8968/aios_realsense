import aios
import time
import threading
import numpy as np


Server_IP_list = []




def main():

    Server_IP_list = aios.broadcast_func()
    
    if Server_IP_list:

        for i in range(len(Server_IP_list)):
            aios.getRoot(Server_IP_list[i])

        print('\n')

        for i in range(len(Server_IP_list)):
            aios.getMotorConfig(Server_IP_list[i], 1)

        print('\n')

        dict = {
            'current_lim' : 40,
            'current_lim_margin' : 5,
            'inverter_temp_limit_lower' : 80,
            'inverter_temp_limit_upper' : 90,
            'requested_current_range' : 60,
            'current_control_bandwidth' : 1000,
        }
        for i in range(len(Server_IP_list)):
            aios.setMotorConfig(dict, Server_IP_list[i], 1)
            aios.saveConfig(Server_IP_list[i])

        print('\n')

        # for i in range(len(Server_IP_list)):
        #     aios.rebootMotorDrive(Server_IP_list[i])
        # time.sleep(2)
        for i in range(len(Server_IP_list)):
            aios.getMotorConfig(Server_IP_list[i], 1)
        print('\n')




if __name__ == '__main__':
    main()
