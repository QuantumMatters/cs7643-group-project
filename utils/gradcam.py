import torch 
from unir.module.SAGAN import ResNetDiscriminator, ResNetGenerator
from unir.dataset.celebA import get_image
from unir.module.corruption import RemovePixel

def load_model(path):
    state_dict = torch.load(path,
                    map_location=torch.device('cpu'))
    model = ResNetDiscriminator(ndf=32)
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    return model


def compute_attributions(algo, inputs, **kwargs):
    '''
    A common function for computing captum attributions
    '''
    return algo.attribute(inputs, **kwargs)

def load_picture(path):

    corr = RemovePixel(0.95)
    img = torch.tensor(get_image(path,108, 108, 64, 64, True, 90))

    measurement = corr.measure(img, device='cpu')

    X_tensor = measurement['measured_sample']
    X_tensor = X_tensor.unsqueeze(0).reshape([1, 3, 64, 64]).type(torch.float32)

    return X_tensor, img, measurement

if __name__ == "__main__":
    celeba_50e = '/content/drive/MyDrive/CS7643-GroupProject/teammates/Kasey/celebA_removePixel_p95_measuredFakeSample2/3/latest_dis.pth'
    celeba_1e = '/content/drive/MyDrive/CS7643-GroupProject/teammates/Kasey/celebA_removePixel_p95_measuredFakeSample/1/latest_dis.pth'

    img_path = '/content/img_align_celeba/img_align_celeba/014965.jpg'
    model = load_model(celeba_50e)
    X_tensor, img, measurement = load_picture(img_path)

    layer = model.block1

    gc_layer = LayerGradCam(model, layer)
    gc_layer_attr = compute_attributions(gc_layer, X_tensor)
    gc_layer_attr_sum = gc_layer_attr.mean(axis=1, keepdim=True)
    attr = gc_layer_attr_sum[0][0].detach().numpy()

    attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
    attr = attr * 0.2 + 0.5
    attr = attr.clip(0.0, 1.0)

    f, axes = plt.subplots(1, 3, sharey=False, figsize=(15, 5))
    plt.suptitle(f'Probability of real corrupted image: {model(X_tensor)[0][0]}', fontsize=15)
    axes[2].imshow(attr)
    axes[1].imshow(measurement['measured_sample'])
    axes[0].imshow(img)