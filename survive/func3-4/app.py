from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
import os
import time
from PIL import Image
from face import *
from style_translation import *

# from test import *
# from query_online import *

app = Flask(__name__)
api = Api(app)

# from extract_cnn_vgg16_keras import VGGNet
# import numpy as np
# import h5py
# import matplotlib.image as mpimg

import shutil
# shutil.copyfile('C:\\1.txt', 'D:\\1.txt')

# global model
global style_result
global face_result
# model = VGGNet()
style_result = {}
face_result = {}

# @app.route('/cla')
# def return_cla():
#     # print(cla)
#     # data = {'name': ['12','214','ds'], 'age': 30}
#     # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

#     return jsonify([cla])


# @app.route('/score')
# def return_score():
#     # print(cla)
#     # data = {'name': ['12','214','ds'], 'age': 30}
#     # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

#     return jsonify([score])


# def query(query, maxres=30, index='/root/CSC4001/featureCNN.h5'):
# 	global score
    	
# 	h5f = h5py.File(index,'r')
# 	# feats = h5f['dataset_1'][:]
# 	feats = h5f['dataset_1'][:]
# 	# print(feats)
# 	imgNames = h5f['dataset_2'][:]
# 	# print(imgNames)
# 	h5f.close()
			
# 	# print("--------------------------------------------------")
# 	# print("               searching starts")
# 	# print("--------------------------------------------------")
		
# 	# read and show query image
# 	queryDir = query
# 	queryImg = Image.open(queryDir)
# 	# queryImg = mpimg.imread(queryDir)

# 	# init VGGNet16 model
# 	import tensorflow as tf

# 	# graph = tf.get_default_graph()
# 	# with graph.as_default():
# 	# model = VGGNet()

# 	# extract query image's feature, compute simlarity score and sort
# 	queryVec = model.extract_feat(queryDir)
# 	scores = np.dot(queryVec, feats.T)
# 	rank_ID = np.argsort(scores)[::-1]
# 	rank_score = scores[rank_ID]

# 	print(rank_score)
# 	for index in range(maxres):
# 		score['score' + str(index+1)] = int(rank_score[index]*100)
# 	#print rank_ID
# 	#print rank_score


# 	# number of top retrieved images to show
# 	# maxres = 5
# 	result = {}
# 	imlist = [imgNames[index].decode('UTF-8')  for i,index in enumerate(rank_ID[0:maxres])]
# 	# result['url'] = imlist[0]
# 	for index, i in enumerate(imlist):
# 		# print('/root/CSC4001/return_img/' + str(index) + '.' + i.split('.')[-1])
# 		shutil.copyfile('/root/CSC4001/image_retrieval/database/' + str(i), '/root/CSC4001/return_img/' + str(index+1) + '.' + i.split('.')[-1])
# 		result['url'+str(index+1)] = i
# 	# print(type(imlist[1]))
# 	# print("top %d images in order are: " %maxres, imlist)
# 	return result






# # TODOS = {
# #     'todo1': {'task': 'build an API'},
# #     'todo2': {'task': '?????'},
# #     'todo3': {'task': 'profit!'},
# # }

# # class TodoSimple(Resource):
# #     def get(self, todo_id):
# #         return {todo_id: TODOS[todo_id]}
# #         # return {'a': TODOS[todo_id]}

# #     def put(self, todo_id):
# #         todos[todo_id] = request.form['data']
# #         return {todo_id: TODOS[todo_id]}

# # api.add_resource(TodoSimple, '/<string:todo_id>')


# @app.route('/json')
# def test_json():
#     # data = {'name': ['12','214','ds'], 'age': 30}
#     data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

#     return jsonify(data)


# @app.route('/func2', methods=['POST'])
# def register():
#     dic = dict(request.form)
#     print(request.method)
#     print(request.form)
#     # print(request.form['name'])
#     # print(request.form.get('name'))
#     # print(request.form.getlist('name'))
#     print(dic)
#     # print(request.form.get('nickname', default='little apple'))
#     return dic



# # 定义路由
# @app.route("/classification", methods=['POST'])
# def get_frame():
#     global cla
#     print()
#     print('='*20 + 'classification' + '='*20)
#     # 接收图片
#     # print(request.data)
#     upload_file = request.files['file']
#     # 获取图片名
#     file_name = upload_file.filename
#     # 文件保存目录（桌面）
#     file_path=r'/root/CSC4001/img'
#     if upload_file:
#         # 地址拼接
#         file_paths = os.path.join(file_path, file_name)
#         print(file_paths)
#         # 保存接收的图片到桌面
#         upload_file.save(file_paths)
#         result = test(file_paths, 0)
#         print(result)
#         print('='*55)
#         print()

#         # # 随便打开一张其他图片作为结果返回，
#         # with open(r'/root/Project/img/1001.jpg', 'rb') as f:
#         #     res = base64.b64encode(f.read())
#         #     return res
#         # time.sleep(30)
#         cla = result
#         # print(cla)
#         return jsonify(result)
#     else:
#         return 'bad'


# # 图像检索
# @app.route("/retrieval", methods=['POST'])
# def retrieval():
#     print()
#     print('='*25 + 'retrieval' + '='*20)
#     # 接收图片
#     # print(request.data)
#     upload_file = request.files['file']
#     # 获取图片名
#     file_name = upload_file.filename
#     # 文件保存目录（桌面）
#     file_path=r'/root/CSC4001/img'
#     if upload_file:
#         # 地址拼接
#         file_paths = os.path.join(file_path, file_name)
#         print(file_paths)
#         # 保存接收的图片到桌面
#         upload_file.save(file_paths)
#         result = query(file_paths, 30)
#         print(result)
#         print('='*55)
#         print()

#         # # 随便打开一张其他图片作为结果返回，
#         # with open(r'/root/Project/img/1001.jpg', 'rb') as f:
#         #     res = base64.b64encode(f.read())
#         #     return res
#         # time.sleep(30)
#         return jsonify(result)
#     else:
#         return 'bad'


#======================================================

def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024

def get_outfile(infile, outfile):
    if outfile:
        return outfile
    dir, suffix = os.path.splitext(infile)
    outfile = '{}-out{}'.format(dir, suffix)
    return outfile

def compress_image(infile, outfile='', mb=150, step=10, quality=80):
    """
    不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    if o_size <= mb:
        return infile
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile, get_size(outfile)


def resize_image(infile, outfile='', x_s=512):
    """
    修改图片尺寸
    :param infile: 图片源文件
    :param outfile: 重设尺寸文件保存地址
    :param x_s: 设置的宽度
    :return:
    """
    im = Image.open(infile)
    x, y = im.size
    y_s = int(y * x_s / x)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    outfile = get_outfile(infile, outfile)
    out.save(outfile)
    return outfile, get_size(outfile)


# 图像检索
@app.route("/style", methods=['POST'])
def style():
    global style_result
    print()
    print('='*25 + 'style' + '='*20)
    # 接收图片
    # print(request.data)
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path=r'/root/CSC4001/data/user_image'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        print(file_paths)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        outfile, size = resize_image(file_paths)
        result = image_style_translation(test_dir=outfile)
        new_path = '/root/CSC4001/results/return/' + str(time.time()).split('.')[0] + '.' + result.split('.')[-1]
        shutil.copyfile(result, new_path)
        style_result['image_url_func3'] = 'http://47.103.45.141:8000/return/' + str(time.time()).split('.')[0] + '.' + result.split('.')[-1]
        print(result)
        print('='*55)
        print()

        # # 随便打开一张其他图片作为结果返回，
        # with open(r'/root/Project/img/1001.jpg', 'rb') as f:
        #     res = base64.b64encode(f.read())
        #     return res
        # time.sleep(30)
        return jsonify(result)
    else:
        return 'bad'




# 图像检索
@app.route("/face", methods=['POST'])
def face():
    global face_result
    print()
    print('='*25 + 'face' + '='*20)
    # 接收图片
    # print(request.data)
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path=r'/root/CSC4001/data/user_image'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        print(file_paths)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        result = face2face(input_image_path=file_paths)
        new_path = '/root/CSC4001/results/return/' + str(time.time()).split('.')[0] + '.' + result.split('.')[-1]
        shutil.copyfile(result, new_path)
        face_result['image_url_func4'] = 'http://47.103.45.141:8000/return/' + str(time.time()).split('.')[0] + '.' + result.split('.')[-1]
        print(result)
        print('='*55)
        print()

        # # 随便打开一张其他图片作为结果返回，
        # with open(r'/root/Project/img/1001.jpg', 'rb') as f:
        #     res = base64.b64encode(f.read())
        #     return res
        # time.sleep(30)
        return jsonify(result)
    else:
        return 'bad'



@app.route('/face_result')
def return_face_result():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify([face_result])


@app.route('/style_result')
def return_style_result():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify([style_result])




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=False)
    # app.run(debug=True)    