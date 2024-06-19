n=int(input())
i=1
str=[]
while i<=n:
    str.append(input())
    i+=1
str.sort()
for s in str:
    print(s)