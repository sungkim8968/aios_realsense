import aios
import time
import threading
import numpy as np
import json

Server_IP_list = []

pos_list_1 = [-124000]
delay_list_1 = [0.3]

pos_list_2 = [1, 2, 3, 4, 0]
delay_list_2 = [0, 0, 0, 0, 1]


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
            aios.setMotorConfig
            for i in range(len(Server_IP_list)):
                enableSuccess = aios.enable(Server_IP_list[i], 1)
            print(enableSuccess)

            if enableSuccess:

                for i in range(len(pos_list_1)):
                    start = time.time()
                    for j in range(len(Server_IP_list)):
                        # aios.setPosition()
                        aios.trapezoidalMove(
                            pos_list_1[i], True, '10.10.10.22', 1)
                    print((time.time() - start)*1)
                    time.sleep(delay_list_1[i])

                #     for i in range(len(pos_list_2)):
                #         start = time.time()
                #         for j in range(len(Server_IP_list)):
                #             aios.trapezoidalMove(pos_list_2[i], True, Server_IP_list[j], 1)
                #             time.sleep( 0.2 )
                #         print((time.time() - start)*1)
                #         time.sleep( delay_list_2[i] )

                # for i in range(800):
                #     start = time.time()
                #     pos = np.sin(i*0.01*np.pi)*1.57
                #     # for j in range(len(Server_IP_list)):
                #     aios.setPosition(pos/6.28*124000, 0, 0,
                #                      False, '10.10.10.27', 1)
                #     time.sleep(0.005)

            #     for i in range(len(Server_IP_list)):
            #         aios.trapezoidalMove(2, True, Server_IP_list[i], 1)
            #     time.sleep( 1 )

            #     for i in range(len(Server_IP_list)):
            #         aios.disable(Server_IP_list[i], 1)


if __name__ == '__main__':
    main()
