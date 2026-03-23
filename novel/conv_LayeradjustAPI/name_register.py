from typing import Type
def getcount(cls:Type):
    cls.count=0
    original_init=cls.__init__
    n=cls.__name__
    def new_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.name=f'{n}-{self.__class__.count}' if self.name is None else self.name
        self.__class__.count+=1
    @classmethod
    def get_count(cls):
        return cls.count
    # def updatename(self):
    #     n=cls.__name__
    #     self.name=f'{n}-{self.__class__.count}'
    cls.__init__=new_init
    cls.get_count=get_count
    return cls