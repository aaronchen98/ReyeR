from network import *
from PIL import Image
import torch
from torchvision import transforms

def change_name(name):
    name = name.replace('_', ' ')
    name = name.split(' ')
    for i in range(len(name)):
        name[i] = name[i].capitalize()
    name = ' '.join(name)
    if name[0].isdigit():
        name = name[0] + ' ' + name[1:]
    return name

def test(img_path, thresh=0.2):

    model = resnet34(pretrained=False)
    state_dict = torch.load('/root/.cache/torch/checkpoints/resnet34-88a5e79d.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()


    input_image = Image.open(img_path).convert('RGB')  # load an image of your choice
    preprocess = transforms.Compose([
        transforms.Resize(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # print(output)

    # The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
    probs = torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
    # print(probs)
    # import matplotlib.pyplot as plt
    import json

    # with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
    #     class_names = json.loads(url.read().decode())

    with open('/root/CSC4001/class_names_500.json') as url:
        class_names = json.loads(url.read())

    # plt.imshow(input_image)
    # plt.grid(False)
    # plt.axis('off')

    def plot_text(thresh=thresh):
        result = {}
        tmp = probs[probs > thresh]
        inds = probs.argsort(descending=True)
        txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
        # for i in inds[0:len(tmp)]:
        for index, i in enumerate(inds[0:6]):
            name = class_names[i]
            name = change_name(name)
            result['name'+str(index+1)] = name
            result['value'+str(index+1)] = int(probs[i].cpu().numpy()*100)

            # result[class_names[i]] = round(float(probs[i].cpu().numpy()), 5)
            txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
        # plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)
        print(txt)

        return result
    result = plot_text()
    # plt.tight_layout()
    # plt.show()

    return result


if __name__ == "__main__":
    test('/root/CSC4001/img/tim.jpeg')