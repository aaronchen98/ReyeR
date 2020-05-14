from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
import os
import time
from PIL import Image
from test import *
# from query_online import *

app = Flask(__name__)
api = Api(app)

from extract_cnn_vgg16_keras import VGGNet
import numpy as np
import h5py
import matplotlib.image as mpimg

import shutil
# shutil.copyfile('C:\\1.txt', 'D:\\1.txt')

global model
global cla
global score
global search
global search2
model = VGGNet()
cla = {}
score = {}
search = {}
search2 = []

@app.route('/cla')
def return_cla():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify([cla])


@app.route('/score')
def return_score():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify([score])


@app.route('/search')
def return_search():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify([search])


@app.route('/search2')
def return_search2():
    # print(cla)
    # data = {'name': ['12','214','ds'], 'age': 30}
    # data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify(search2)

def query(query, maxres=30, index='/root/CSC4001/featureCNN2.h5'):
	# global score
	global search	
	global search2
	search2 = []
	result = {}

	h5f = h5py.File(index,'r')
	# feats = h5f['dataset_1'][:]
	feats = h5f['dataset_1'][:]
	# print(feats)
	imgNames = h5f['dataset_2'][:]
	# print(imgNames)
	h5f.close()
			
	# print("--------------------------------------------------")
	# print("               searching starts")
	# print("--------------------------------------------------")
		
	# read and show query image
	queryDir = query
	queryImg = Image.open(queryDir)
	# queryImg = mpimg.imread(queryDir)

	# init VGGNet16 model
	import tensorflow as tf

	# graph = tf.get_default_graph()
	# with graph.as_default():
	# model = VGGNet()

	# extract query image's feature, compute simlarity score and sort
	queryVec = model.extract_feat(queryDir)
	scores = np.dot(queryVec, feats.T)
	rank_ID = np.argsort(scores)[::-1]
	rank_score = scores[rank_ID]

	print(rank_score)
	for index in range(maxres):
		result['score' + str(index+1)] = int(rank_score[index]*100)
		search2.append({'score':int(rank_score[index]*100)})
	#print rank_ID
	#print rank_score


	# number of top retrieved images to show
	# maxres = 5
	imlist = [imgNames[index].decode('UTF-8')  for i,index in enumerate(rank_ID[0:maxres])]
	# result['url'] = imlist[0]
	for index, i in enumerate(imlist):
		# print('/root/CSC4001/return_img/' + str(index) + '.' + i.split('.')[-1])
		shutil.copyfile('/root/CSC4001/image_retrieval/database/' + str(i), '/root/CSC4001/return_img/' + str(index+1) + '.' + i.split('.')[-1])
		result['url'+str(index+1)] = 'http://39.102.65.214:8000/image_retrieval/database/' + i
		search2[index]['image_url'] = 'http://39.102.65.214:8000/image_retrieval/database/' + i
	# print(type(imlist[1]))
	# print("top %d images in order are: " %maxres, imlist)
	search = result
	return result






# TODOS = {
#     'todo1': {'task': 'build an API'},
#     'todo2': {'task': '?????'},
#     'todo3': {'task': 'profit!'},
# }

# class TodoSimple(Resource):
#     def get(self, todo_id):
#         return {todo_id: TODOS[todo_id]}
#         # return {'a': TODOS[todo_id]}

#     def put(self, todo_id):
#         todos[todo_id] = request.form['data']
#         return {todo_id: TODOS[todo_id]}

# api.add_resource(TodoSimple, '/<string:todo_id>')


@app.route('/json')
def test_json():
    # data = {'name': ['12','214','ds'], 'age': 30}
    data = {'url': ['http://39.102.65.214:8000/2375001.jpg', 'http://39.102.65.214:8000/573001.jpg', 'http://39.102.65.214:8000/1091000.jpg', 'http://39.102.65.214:8000/2283000.jpg', 'http://39.102.65.214:8000/1498002.jpg', 'http://39.102.65.214:8000/558002.jpg', 'http://39.102.65.214:8000/860000.jpg', 'http://39.102.65.214:8000/2170002.jpg', 'http://39.102.65.214:8000/558001.jpg', 'http://39.102.65.214:8000/2522000.jpg']}

    return jsonify(data)


@app.route('/func2', methods=['POST'])
def register():
    dic = dict(request.form)
    print(request.method)
    print(request.form)
    # print(request.form['name'])
    # print(request.form.get('name'))
    # print(request.form.getlist('name'))
    print(dic)
    # print(request.form.get('nickname', default='little apple'))
    return dic



# 定义路由
@app.route("/classification", methods=['POST'])
def get_frame():
    global cla
    print()
    print('='*20 + 'classification' + '='*20)
    # 接收图片
    # print(request.data)
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path=r'/root/CSC4001/img'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        print(file_paths)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        result = test(file_paths, 0)
        print(result)
        print('='*55)
        print()

        # # 随便打开一张其他图片作为结果返回，
        # with open(r'/root/Project/img/1001.jpg', 'rb') as f:
        #     res = base64.b64encode(f.read())
        #     return res
        # time.sleep(30)
        cla = result
        # print(cla)
        return jsonify(result)
    else:
        return 'bad'


# 图像检索
@app.route("/retrieval", methods=['POST'])
def retrieval():
    print()
    print('='*25 + 'retrieval' + '='*20)
    # 接收图片
    # print(request.data)
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path=r'/root/CSC4001/img'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        print(file_paths)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        result = query(file_paths, 30)
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




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=False)
    # app.run(debug=True)    