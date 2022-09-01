a = []
b = []
for j in range(50):
    b.append(j)
for i in range(100):
    a.append(i)
c= [a,b]
filename = open('a.txt', 'w')
for value in c:
     filename.write(str(value))
filename.close()
# 读取
f= open("a.txt","r")
d = f.read()
f.close()