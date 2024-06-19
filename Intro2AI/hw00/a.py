s1,s2=input().split()
if len(s1) >= len(s2):
    s=s1;s1=s2;s2=s
print(s2+s1)