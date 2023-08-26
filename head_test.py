import aios
import time
import threading
import numpy as np
import json


Server_IP_list = []


def main():

    Server_IP_list = aios.broadcast_func()

    if Server_IP_list:

        encoderIsReady = True
        for i in range(len(Server_IP_list)):
            if (not aios.encoderIsReady(Server_IP_list[i], 1)):
                encoderIsReady = False

        print('\n')

        if encoderIsReady:
            for i in range(len(Server_IP_list)):
                aios.getRoot(Server_IP_list[i])

            print('\n')
            # aios.setMotorConfig
            for i in range(len(Server_IP_list)):
                enableSuccess = aios.enable(Server_IP_list[i], 1)
            print(enableSuccess)

            if enableSuccess:

                while (1):
                    # aios.setPosition(10/360 * 31, 0, 0,
                    #                  False, '10.10.20.93', 1)
                    pos, vel, cur = aios.getCVP('10.10.20.93', 1)
                    print("pos", pos)
                    # aios.setPosition(qs_right[i][1]/6.28 * 204000, 0, 0, False, '10.10.10.12', 1)
                    # aios.setPosition(-qs_right[i][2]/6.28 * 204000, 0, 0, False, '10.10.10.13', 1)
                    # aios.setPosition(-qs_right[i][3]/6.28 * 124000, 0, 0, False, '10.10.10.14', 1)
                    # aios.setPosition(-qs_right[i][4]/6.28 * 124000, 0, 0, False, '10.10.10.15', 1)
                    # aios.setPosition(-qs_right[i][5]/6.28 * 124000, 0, 0, False, '10.10.10.16', 1)
                    # aios.setPosition(-qs_right[i][6]/6.28 * 124000, 0, 0, False, '10.10.10.17', 1)
                    # aios.setPosition(-qs_left[i][0]/6.28 * 204000, 0, 0, False, '10.10.10.21', 1)
                    # aios.setPosition(qs_left[i][1]/6.28 * 204000, 0, 0, False, '10.10.10.22', 1)
                    # aios.setPosition(-qs_left[i][2]/6.28 * 204000, 0, 0, False, '10.10.10.23', 1)
                    # aios.setPosition(-qs_left[i][3]/6.28 * 124000, 0, 0, False, '10.10.10.24', 1)
                    # aios.setPosition(-qs_left[i][4]/6.28 * 124000, 0, 0, False, '10.10.10.25', 1)
                    # aios.setPosition(-qs_left[i][5]/6.28 * 124000, 0, 0, False, '10.10.10.26', 1)
                    # aios.setPosition(-qs_left[i][6]/6.28 *124000, 0, 0, False, '10.10.10.27', 1)

                    time.sleep(0.005)


if __name__ == '__main__':
    main()
