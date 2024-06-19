n=(int)(input())
str = input()
a = str.split(" ")
b = []
for i in range(0,n):
    b.append(int(a.pop()))
b.sort()
if n%2==1:
    print(b[n//2])
elif (b[n//2]+b[n//2-1])%2==1:
    print(round((b[n//2]+b[n//2-1])/2,1))
else:
    print((b[n//2]+b[n//2-1])//2)