f = open("bank/output548287704427.dat", "r")
a = f.readlines()
f.close()

aa = open('bank/comments_iso.dat', 'a')

for i, j in enumerate(a):
    if (j[0:6] == '# Zini'):
        aa.close()
        aa = open('bank/age_{:.2f}_Gyr_MH_{}.dat'.format(10.**(float(a[i+1].split()[2])-9), a[i+1].split()[1][0:5]), 'a')
        print(j, file=aa, end='')
    else:
        print(j, file=aa, end='')
aa.close()
