n,k = map(int,input().split())
p = 200
i, s = 1, 0
while i<=20:
    s += n
    if s >= p: break
    p = p * (1+k/100)
    i += 1
if s>=p :print(i)
else:print('Impossible')