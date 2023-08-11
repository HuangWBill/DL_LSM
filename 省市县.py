import requests
import xlrd
from xlutils.copy import copy

#导入需要读取Excel表格的路径
book_name_xls = 'C:/Users/dell/Desktop/1(1).xls'
data = xlrd.open_workbook(r'C:/Users/dell/Desktop/1(1).xls')
table = data.sheets()[0]
# 创建一个空列表，存储Excel的数据
jingdu = []
weidu=[]
sheng=[]
shi=[]
xian=[]

# 将excel表格内容导入到tables列表中
def import_excel(excel):
    for rown in range(excel.nrows):
        weidu.append(table.cell_value(rown, 1))
        jingdu.append(table.cell_value(rown, 2))

#将excel表格的内容导入到列表中
import_excel(table)
print(jingdu)
print(weidu)
def write_excel_xls_append(path, value):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    rows_old = 3  # 获取表格中已存在的数据的列数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for j in range(0,len(value)):
        new_worksheet.write(i, j+rows_old, value[j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")

def locatebyLatLng(m):
     # 根据经纬度查询地址
    # key = 'GjG3XAdmywz7CyETWqHwIuEC6ZExY6QT'
    # baiduUrl = "http://api.map.baidu.com/reverse_geocoding/v3/?ak=xl1awZfyMy3WGGSUsCmwSzlBLjZEETZQ&output=json&coordtype=wgs84ll&location="+str(lat)+","+ str(lng)
    items = {'location': m, 'ak': 'GjG3XAdmywz7CyETWqHwIuEC6ZExY6QT', 'output': 'json'}
    res = requests.get(url='http://api.map.baidu.com/geocoder/v2/', params=items)
    # res = requests.get(baiduUrl)
    result = res.json()
    print('--------------------------------------------')
    # result = result['result']['formatted_address'] + ',' + result['result']['sematic_description']
    province = result['result']['addressComponent']["province"]
    city = result['result']['addressComponent']['city']
    district = result['result']['addressComponent']["district"]
    return province,city,district
for i in range(len(jingdu)):
    province,city,district = locatebyLatLng(str(jingdu[i])+','+str(weidu[i]))
    sheng.append(province)
    shi.append(city)
    xian.append(district)
    value= [province, city, district]
    write_excel_xls_append(book_name_xls, value)
    print('第'+str(i)+'个添加完成！')
print('------------------------over---------------------')


