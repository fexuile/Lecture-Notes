str = input().split(" ")
n=len(str);n-=1
while n>0:
    print(str[n],end=' ')
    n-=1
print(str[0])