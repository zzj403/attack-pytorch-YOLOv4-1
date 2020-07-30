import os

file_list = os.listdir('select1000')
file_list.sort()

for file_name in file_list:

    srcFile = os.path.join('select1000', file_name)
    new_name = file_name
    new_name = new_name.replace('select1000', '')
    # new_name = new_name.strip('select1000')
    new_name = new_name.rjust(8, '0')
    dstFile = os.path.join('select1000', new_name)
    try:
        os.rename(srcFile,dstFile)
    except Exception as e:
        print(e)
        print('rename file fail\r\n')
    else:
        print('rename file success\r\n')
    print()

