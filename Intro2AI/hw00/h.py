a = input()
b = input()
if len(a) > len(b):
    c=a;a=b;b=c;
if len(a) == len(b):
    if a==b:print(a,'is substring of',b)
    else:print('No substring')
else:
    flag2=0
    for i in range(0,len(b)-len(a)+1):
        flag = 1
        for j in range(0,len(a)):
            if a[j] != b[i+j]: flag = 0
        if flag : flag2=1;break
    if flag2: print(a,'is substring of',b)
    else:print('No substring')
        