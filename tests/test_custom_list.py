from typing import Optional, List, Union, Any, Sequence


class MyList:
    def __init__(self, elements: Optional[List] = None):
        if elements:
            self.elements = elements
        else:
            self.elements = []

    def __getitem__(self, item: Union[int, Sequence[int]]) -> Any:
        if isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, tuple):
            elem = self.elements[item[0]]
            for index in range(1, len(item)):
                elem = elem[index]
            return elem
        else:
            raise TypeError(f"Invalid item type: {type(item)}")

    def __str__(self):
        content = ", ".join(map(str, self.elements))
        return content


if __name__ == "__main__":
    l1 = MyList([
        [1, 2, 3],
        [4, 5, 6],
    ])
    print(l1)
    # e = l1[1, 2]
    # print(e)
