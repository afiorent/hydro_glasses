#code partially imported from enrdrigo "transport with densities" github.
import numpy as np
import pickle as pk
import time
import h5py
import os
import logging
import warnings
from multiprocessing import Pool


def read_dump(root, filename, Np, ntry,ncol=5):
    with open(root + filename, 'r') as f:
        for indexx, lines in enumerate(f):

            linesplit = []

            for i in lines.split(' '):

                if i != '': linesplit.append(i)

            if len(linesplit) > 2 and linesplit[1] == 'ATOMS':
                dickeys=[]
                for i in linesplit[2:]:
                    if i =='\n': continue
                    dickeys.append(str(i))
                print('you dumped this quantities:\n', dickeys)
                break

    with open(root + filename, 'r') as f:

        if os.path.exists(root + 'dump.h5'):
            with h5py.File(root + 'dump.h5', 'r') as dump:
                snap = [[] for i in range(dump['data'].len())]

            lenght = len(snap)
            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            dump = h5py.File(root + 'dump.h5', 'a')
            lenght = 0

        dump = h5py.File(root + 'dump.h5', 'a')
        d = []
        start = time.time()

        logging.info(str(f.name))
        logging.info(str(dump.keys()))
        start0 = time.time()

        for index, line in enumerate(f):
            if int((index+1) % 1.0e7) == 0:
                dump.close()
                dump = h5py.File(root + 'dump.h5', 'a')
                logging.info(str(dump['data'].len()))
                logging.info(str(time.time()-start0))
                start0=time.time()

            if index < lenght * (Np + 9):
                continue

            linesplit = []

            for i in line.split(' '):

                if i != '': linesplit.append(i)

            if len(linesplit) != len(dickeys):
                continue
            dlist = [float(linesplit[i]) for i in range(len(dickeys))]
            d.append(dlist)

            if (index + 1) % (Np + 9) == 0:

                if len(d) == 0:
                    print('END READ FILE')
                    print('got ' + str((index + 1) // (Np + 9)) + ' snapshot')
                    dump.close()
                    return

                elif len(d) != Np:

                    print(len(d))
                    print('STOP: THE SNAPSHOT ' + str((index + 1) // (Np + 9)) + ' DOES NOT HAVE ALL THE PARTICLES')
                    print('got ' + '' + ' snapshot')
                    dump.close()
                    return

                datisnap = np.array(d)
                d = []
                if index == Np + 8:
                    dump.create_dataset('data', data=datisnap[np.newaxis, :, :], compression="gzip", chunks=True,
                            maxshape=(None,datisnap.shape[0], datisnap.shape[1]))
                else:
                    dump['data'].resize((dump['data'].shape[0] + 1), axis=0)
                    dump['data'][-1] = datisnap

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9) / 3)
                    print('done')
                    print('END READ. NO MORE DATA TO LOAD. SEE NTRY')
                    dump.close()
                    return

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9))
                    print('done')
                    print('elapsed time: ', time.time() - start)
                    print('END READ NTRY')
                    dump.close()
                    return

        print('number of total snapshots is', (index + 1) // (Np + 9))
        print('done')
        print('elapsed time: ', time.time() - start)
        print('END READ FILE GOOD')
        dump.close()
        return


