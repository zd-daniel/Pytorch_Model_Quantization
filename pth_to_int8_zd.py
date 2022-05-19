# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 17:30
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : pth_to_int8_zd.py
# @Software: PyCharm


import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm


class PoseEstimation(nn.Module):
    def __init__(self, model):
        super(PoseEstimation, self).__init__()

        self.quant = torch.quantization.QuantStub()

        ########################## 根据模型需要重写 ############################
        self.layer0 = model.model.model[0]
        self.layer1 = model.model.model[1]
        self.layer2 = model.model.model[2]
        self.layer3 = model.model.model[3]
        self.layer4 = model.model.model[4]
        self.layer5 = model.model.model[5]
        self.layer6 = model.model.model[6]
        self.layer7 = model.model.model[7]
        self.layer8 = model.model.model[8]
        self.layer9 = model.model.model[9]
        self.layer10 = model.model.model[10]
        self.layer11 = model.model.model[11]
        self.layer12 = model.model.model[12]
        self.layer13 = model.model.model[13]
        self.layer14 = model.model.model[14]
        self.layer15 = model.model.model[15]
        self.layer16 = model.model.model[16]
        self.layer17 = model.model.model[17]
        self.layer18 = model.model.model[18]
        self.layer19 = model.model.model[19]
        self.layer20 = model.model.model[20]
        self.layer21 = model.model.model[21]
        self.layer22 = model.model.model[22]
        self.layer23 = model.model.model[23]
        self.layer24 = model.model.model[24]
        ########################## 根据模型需要重写 ############################

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        ################## 根据模型需要重写 ############################
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output_80_80 = x
        x = self.layer5(x)
        x = self.layer6(x)
        output_40_40 = x
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        output_20_20 = x

        x = self.layer10(x)
        output_20_20 = x
        x = self.layer11(x)
        x = self.layer12([x, output_40_40])
        x = self.layer13(x)
        x = self.layer14(x)
        output_40_40 = x

        x = self.layer15(x)
        x = self.layer16([x, output_80_80])
        x = self.layer17(x)
        output_80_80 = x

        x = self.layer18(x)
        x = self.layer19([x, output_40_40])
        x = self.layer20(x)
        output_40_40 = x

        x = self.layer21(x)
        x = self.layer22([x, output_20_20])
        x = self.layer23(x)
        output_20_20 = x

        x, _ = self.layer24([output_80_80, output_40_40, output_20_20])
        ################## 根据模型需要重写 ############################

        # output = self.dequant(x)  # YOLOv5的layer24层中Detect已反量化，这里不再需要
        output = x
        return output


class Pth2Int8:
    '''
    pytorch训练好pth模型，利用pytorch中quantization量化为int8/uint8参数保存，对称或非对称量化将带来scale和zero_point
    '''
    def __init__(self, model=None, checkpoint=None) -> None:
        '''
        :param model: pth模型
        :param checkpoint: 模型权重文件
        '''
        assert model is not None, '未加载已训练好的模型'
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)  # 读取预训练模型
            model.load_state_dict(state_dict)
        model = PoseEstimation(model)

        # model must be set to eval mode for static quantization logic to work
        self.model_fp = model.float().eval()

    def do_quant(self, save_path='./quant_model.pth', train_data_dir='./data', img_size=640):
        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference and
        # 'qnnpack' for mobile inference. Other quantization configurations such
        # as selecting symmetric or assymetric quantization and MinMax or L2Norm
        # calibration techniques can be specified here.
        self.model_fp.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        model_fp_prepared = torch.quantization.prepare(self.model_fp)

        # calibrate the prepared model to determine quantization parameters for activations
        # in a real world setting, the calibration would be done with a representative dataset
        self.__evaluate(model_fp_prepared, train_data_dir, img_size)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        model_int8 = torch.quantization.convert(model_fp_prepared)

        print('模型量化完成(Int8)，保存为{}'.format(save_path))
        # save model
        torch.save(model_int8.state_dict(), save_path)

    def __evaluate(self, model, train_data_dir, img_size):
        '''
        :param model:
        :param train_data_dir: 用于统计feature map中scale和zero_point，选择训练集即可
        :param img_size:
        :return:
        '''
        scale_search = [0.5, 1.0, 1.5, 2.0]
        param_stride = 32

        # Predict pictures
        list_dir = os.walk(train_data_dir)
        print('模型转换中...\n')
        for root, dirs, files in list_dir:
            for f in tqdm(files):
                test_image = os.path.join(root, f)
                img_ori = cv2.imread(test_image)  # B,G,R order

                multiplier = [scale * img_size / img_ori.shape[0] for scale in scale_search]

                for i, scale in enumerate(multiplier):
                    h = int(img_ori.shape[0] * scale)
                    w = int(img_ori.shape[1] * scale)
                    pad_h = 0 if (h % param_stride == 0) else param_stride - (h % param_stride)
                    pad_w = 0 if (w % param_stride == 0) else param_stride - (w % param_stride)
                    new_h = h + pad_h
                    new_w = w + pad_w

                    img_test = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    img_test_pad, pad = self.__pad_right_down_corner(img_test, param_stride, param_stride)
                    img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 255

                    feed = torch.from_numpy(img_test_pad)
                    output = model(feed)

    def __pad_right_down_corner(self, img, stride, pad_value):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad

    def __del__(self):
        pass


class InferenceFromPthInt8:
    def __init__(self, model=None, quant_checkpoint=None):
        '''
        :param model: 加载模型
        :param quant_checkpoint: 模型权重文件
        '''
        assert model is not None, '未加载模型'
        assert quant_checkpoint is not None, '未加载Int8模型'

        # Load int8 model
        state_dict = torch.load(quant_checkpoint)
        model_fp32 = PoseEstimation(model)
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        model_int8 = torch.quantization.convert(model_fp32_prepared)
        model_int8.load_state_dict(state_dict)

        self.model = model_int8
        self.model.eval()

    def __call__(self, input):
        return self.model(input)

    def __del__(self):
        pass


if __name__ == '__main__':
    from models.common import DetectMultiBackend
    weights = 'runs/train/exp/weights/best.pt'
    device = torch.device('cpu')
    dnn = False
    data = 'data/coco128.yaml'
    model_fp32 = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)

    ################################# 模型量化为Int8 ########################
    # quant_model = Pth2Int8(model_fp32)
    # quant_model.do_quant(train_data_dir='datasets/coco128/images/train2017_')
    ################################# 模型量化为Int8 ########################

    ################################# Int8量化模型前向推断 ########################
    quant_model_path = './quant_model.pth'
    infer_model = InferenceFromPthInt8(model_fp32, quant_model_path)

    img_path = 'D:/Work/Work_Python/yolov5-master/datasets/coco128/images/train2017/FLIR_04541.jpeg'
    image = cv2.imread(img_path)
    image_show = np.copy(image)
    image = torch.from_numpy(image[None, ...]).permute(0, 3, 1, 2).float().cpu()
    image /= 255

    import time
    start_time = time.time()
    pred = infer_model(image)
    # pred = model_fp32(image)
    print(time.time() - start_time)
    print(pred)
    print(pred.shape)

    from utils.general import non_max_suppression
    conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.028, 0.045, None, False, 1000
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(pred)

    from utils.plots import Colors
    color_set = Colors()
    for p in pred[0]:
        bbox, conf, cls = p[:4], p[4], p[5]
        bbox, conf, cls = bbox.detach().round().int().numpy(), conf.detach().numpy(), cls.detach().numpy()
        cv2.rectangle(image_show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_set(cls), 2, 2)
        cv2.putText(image_show, '%.2f' % (100 * conf), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, color_set(cls), 2)
    cv2.imshow('image', image_show)
    cv2.waitKey(0)

    # FP32+GPU 13ms
    # FP32+CPU 226ms
    # int8+cpu 57ms

    # FLIR 车载红外数据集
    # FP32：P=0.834; R=0.541; mAP@.5=0.621; mAP@.5:.95=0.324
    # int8：P=0.803; R=0.534; mAP@.5=0.609; mAP@.5:.95=0.284
