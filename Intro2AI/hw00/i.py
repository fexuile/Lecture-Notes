str = input()
n = len(str)
flag = 1
for i in range(0,n//2):
    if str[i]!=str[n-i-1]:flag=0
if flag:print(1)
else : print(0)