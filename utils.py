from keras import backend as K

def recall_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 0 if c3 == 0 else c1/c3

def precision_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 0 if c3 == 0 else c1/c2

def f1_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = c1 / c2
    recall = 0 if c3 == 0 else c1 / c3
    f1_score = 2 * (precision * recall) / (precision + recall)
    return 0 if c3 ==0 else f1_score

# implement fixed length link from list
#   insert_index record the insert position of list
#   value is the value to insert
#   lst is a list to implements link
#   list_size is the size of lst
# for example:
#   lst = [0,0,0]
#   insert_index = 0
#   lst,insert_index = insert_into_listlikelink(lst,3,insert_index,1)
#   lst,insert_index = insert_into_listlikelink(lst,3,insert_index,2)
#   lst,insert_index = insert_into_listlikelink(lst,3,insert_index,3)
#           lst = [1,2,3]
#   lst,insert_index = insert_into_listlikelink(lst,3,insert_index,4)
#           lst = [4,2,3]
#   lst,insert_index = insert_into_listlikelink(lst,3,insert_index,6)
#           lst = [4,5,3]
def insert_into_listlikelink(lst, list_size, insert_index, value):
    lst[insert_index] = value
    insert_index += 1
    insert_index = insert_index % list_size
    return lst,insert_index

# reduction ordered list from fixed length link
#   insert_index record the insert position of list
#   lst is a list to implements link
# return
#   ordered list from most recently to earliest(LIFO,last in first out)
# for example:
#   lst = [4,5,3]
#   insert_index = 2
#   lst = read_from_listlikelink(lst,insert_index)
#           lst = [5,4,3]
def read_from_listlikelink(lst,insert_index):
    lst_order_before = lst[0:insert_index]
    lst_order_after = lst[insert_index:]
    lst_order_before.reverse()
    lst_order_after.reverse()
    lst_order = lst_order_before + lst_order_after
    return lst_order