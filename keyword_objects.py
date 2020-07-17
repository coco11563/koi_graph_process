import torch as torch
import sys

class StackUnderflow(ValueError): # 栈下溢(空栈访问)
    pass

class SStack:
    def __init__(self):
        self._elems = [] # 使用list存储栈元素

    def is_empty(self):
        return self._elems == []

    def push(self, elem):
        self._elems.append(elem)

    def pop(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.pop()")
        return self._elems.pop()

    def top(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.top()")
        return self._elems[-1]

class field :
    def __init__(self, ros, apply_id):
        self.ros = ros
        self.apply_id = apply_id

    def __hash__(self):
        if self.ros == None:
            return hash(self.apply_id)
        else:
            return hash(self.ros + self.apply_id)

    def __eq__(self, other):
        if isinstance(other, keyword):
            return other.__hash__() == self.__hash__()
        else:
            return False

class embedding() :
    def __init__(self, vector):
        self.tensor = torch.tensor(vector)

class keyword :
    def __init__(self, name, ros, apply_id, frequent = 0, frequent_all = 0):
        self.name = name.replace("？","").replace(" ", "")
        self.ros = ros
        self.apply_id = apply_id
        self.frequent = frequent
        self.frequent_all = frequent_all
    def __hash__(self):
        if self.ros == None :
            return hash(self.name + self.apply_id)
        else:
            return hash(self.name + self.ros + self.apply_id)
    def __eq__(self, other):
        if isinstance(other, keyword) :
            return other.__hash__() == self.__hash__()
        else: return False
    def __str__(self):
        return self.apply_id + " " + self.ros + " " + self.name

class application(object):
    def __init__(self, applyid, title,abstract, keyword,research_field):
        self.title = title.replace("？","").replace(" ", "")
        self.abstract = abstract.replace("？","").replace(" ", "")
        self.research_field = research_field
        self.applyid = applyid.replace("？","").replace(" ", "")
        self.keyword = keyword.replace("？","").replace(" ", "").split("；")
    def __hash__(self):
        if self.research_field == None :
            return  (self.title + self.applyid).__hash__()
        else :
            return (self.title + self.applyid + self.research_field).__hash__()
class apply_id_tree :
    def __init__(self):
        self.root = dict()

    def add_(self, root : dict, leaf, head: str):
        this_id = head + leaf[0:2]
        if len(leaf) > 1 :
            if not root.__contains__(this_id) :
                root.update({this_id: dict()})
            self.add_(root.get(this_id), leaf[2:], head + leaf[0:2])

    def add(self, id : str) : #to do
        head = id[0]
        if self.root.__contains__(head) :
            dic = self.root.get(head)
            self.add_(dic, id[1:], head)
        else:
            self.root.update({head: dict()})
            dic = self.root.get(head)
            self.add_(dic, id[1:], head)

    def get_(self, id, root : dict, head : str, ret_li : list) :
        this_id = head + id[0:2]
        if root.__contains__(this_id) : # 有值
            dic : dict = root.get(this_id)
            if len(id) == 2 : # last two 这就是需要的值
                # print(dic)
                ret_li.append(this_id)
                for elem in dic.keys():
                    ret_li.append(elem)
                # add other below
                value = SStack()
                for elem in dic.values():
                    # print(elem)
                    value.push(elem)
                while(not value.is_empty()) :
                    elems = value.pop()
                    # print(elems)
                    for elem in elems.values():
                        value.push(elem)
                    for keys in elems.keys():
                        ret_li.append(keys)
            else :
                self.get_(id[2:], dic, this_id, ret_li)

    def get(self, id) :
        if len(id) == 1 :
            return self.get_all_by_id(id)
        head = id[0]
        ret_li = []
        if self.root.__contains__(head) :
            dic = self.root.get(head)
            # print(dic)
            self.get_(id[1:], dic, head, ret_li)
        return ret_li
    # def get(self, apply_id, ros) :

    def get_all_(self, root, set) :
        for elem in root.keys() :
            set.add(elem)
            self.get_all_(root.get(elem), set)

    def get_all(self) :
        root = self.root
        all_apply_id = set()
        for value in root.values() :
            self.get_all_(value, all_apply_id)
        return all_apply_id

    def count(self) :
        return len(self.get_all())

    def get_all_by_id_(self, root) :
        all_apply_id = set()
        for key in root.keys() :
            all_apply_id.add(key)
        for value in root.values() :
            self.get_all_(value, all_apply_id)
        return all_apply_id

    def get_all_by_id(self, id):
        root = self.root[id]
        return list(self.get_all_by_id_(root))

def main():
    t = apply_id_tree()
    t.add("A010101")
    t.add("A020102")
    t.add("B010101")
    t.add("A0102")
    t.add("A0202")
    # print(t.get("A"))
    # print(t.get_all())
    print(t.root.keys())
    for i in t.root.keys() :
        print(t.root[i]) # new root
    print(t.count())

    # print(t.root)
    # for elem in t.get("A010101") :
    #     print(elem)
    # for elem in t.get_all() :
    #     print(elem)

if __name__ == '__main__':
    main()
    # print(__name__)


