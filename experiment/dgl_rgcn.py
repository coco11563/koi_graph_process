import scipy.io
import urllib.request

data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))

