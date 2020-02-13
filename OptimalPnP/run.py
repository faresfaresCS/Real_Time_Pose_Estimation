import numpy as np
import os
import time


# Optimal PnP
def optimalPnP(points3D, lines):
    with open('OMat.txt', 'w') as f:
        f.write("%s 9 30\n" % len(points3D))
        for i in range(len(points3D)):
            for item in lines[i]:
                f.write("%s " % item)
            f.write("0 0 0 ")
            for item in points3D[i]:
                f.write("%s " % item)
            f.write("\n")
        f.write("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n")
        for item in points3D:
            f.write("1 ")

    # for tests:
    start_time = time.time()
    os.system("sudo ./pnp_exe OMat.txt")
    print("--- %s seconds for Optimal PnP on full data ---" % (time.time() - start_time))

    with open('OPnP_R.txt', 'r') as f:
        rv1 = [[float(num) for num in line.split(',')] for line in f]
    with open('OPnP_t.txt', 'r') as f:
        t = f.read().split(',')
    tv1 = []
    for el in t:
        tv1.append(float(el))

    tv = np.array(tv1);
    rv = np.array(rv1);
    return rv, tv


def generatelines(pixels,K):
    alllines=[]
    # if K.shape[0]!=3 or K.shape[1]!=3:
    #    sys.exit('camera matrice is not valid it should be 3x3')
    for x, y in pixels:
        x=np.float64(x)
        y=np.float64(y)
        z = 1
        npoint = np.asarray((x, y, z))
        line=np.matmul(np.linalg.inv(K),npoint)
        line=line/np.linalg.norm(line)
        # print("**********")
        # print(np.linalg.norm(line))
        # print("-***************------")
        alllines.append(line)
    return alllines


def main():
    # record_cam()

        # test = input()
        point = []
        line = []
        K = [ [1578.47533, 0, 320], [0,
       1771.81, 240], [0, 0, 1] ]
        with open('2dpoints.txt', 'r') as f:
            points2d = [[float(num) for num in line.split(' ')] for line in f]
        with open('3dpoints.txt', 'r') as f:
            points3d = [[float(num) for num in line.split(' ')] for line in f]
        lines = generatelines(points2d, K)
        R,t = optimalPnP(points3d, lines)


if __name__ == "__main__":
    main()



