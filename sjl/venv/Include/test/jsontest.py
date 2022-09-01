import json
class Test:
    def __init__(self,name,feeling):
        self.name = name
        self.feeling = feeling
    def prt(self):
        print(self)
        print(self.name)


t = Test('hhh','zzz')
t.prt()
obj = {
    'name' :t.name,
    'feeling':t.feeling
}
with open('./obj.json','w',encoding='utf-8') as fObj:
    json.dump(obj,fObj,ensure_ascii=False)
with open('./obj.json','r',encoding='utf-8') as fObj:
    s=json.load(fObj)
t = Test(s['name'],s['feeling'])
t.prt()