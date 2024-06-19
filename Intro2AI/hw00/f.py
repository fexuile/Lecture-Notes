def max(a,b):
    if a>b:return a
    return b
def min(a,b):
    if a<b:return a
    return b
n=6
str = input().split(" ")
mx = 0
mn = 110
for i in range(0,n):
    x = int(str.pop())
    if(x%2==0):mn=min(mn,x)
    else :mx=max(mx,x)
print(abs(mx-mn))