import torch
from dataloader import MyDataset
from predict import config
from predict import model_cls

idx2label = config.idx2label
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
textcls_model = config.textcls_model

def data_pro(contents):
    texts = []
    others = []
    for i,content in enumerate(contents):
        content = content.replace('…', '').replace('..','')
        text = ''
        for n in range(0, len(content)-1):
#             if '\u4e00' <= content[n] <= '\u9fff' or content[n] in '。？！，；：、‘“”’/（）《》.0123456789.%':
            if '\u4e00' <= content[n] <= '\u9fff' or content[n] in '@$*&!,。？！，；：:-_‘“”’/（）《》.0123456789.% '\
                    or '\u0041' <= content[n] <= '\u005A' or '\u0061' <= content[n] <= '\u007A':
                text += content[n]
        texts.append(text)
        if len(text)<50:
            others.append(i)
    return texts,others


def text_classification(sentence_tests):
    sentence_tests ,others = data_pro(sentence_tests)
    #GPU训练保存全模型，并用GPU加载全模型
    loadnn = model_cls.Bert_Blend_CNN()
#     loadnn = torch.load(textcls_model)

    #GPU训练保存模型参数，并用CPU加载参数
#     state_dict = torch.load(textcls_model,map_location=device)
#     loadnn.load_state_dict(state_dict=state_dict)
#     loadnn.to(device)
#     loadnn.eval()
    
    #GPU训练保存模型参数，并用GPU加载参数
    state_dict = torch.load(textcls_model,map_location=device)
    loadnn.load_state_dict(state_dict=state_dict) 
    loadnn.to(device)
    loadnn.eval()
    
    label_pres = []
    pre_texts = []
    result_dict = {}

    with torch.no_grad():
        test = MyDataset(sentence_tests, labels=None, with_labels=False)
        print(len(test))
        for i,test_text in enumerate(sentence_tests):
            x = test.__getitem__(i)
            x = tuple(p.unsqueeze(0).to(device) for p in x)
            # print(x[0], x[1], x[2])
            pred = loadnn([x[0], x[1], x[2]])
            # print("11111111111111",pred.shape)
            pred = pred.data.max(dim=1, keepdim=True)[1]
            label_pre = int(pred[0][0])
            label_pres.append(label_pre)
            if i in others:
                label_pre = 6
            pre_text = idx2label[label_pre]
            pre_texts.append(pre_text)
    result_dict['label'] = label_pres
    result_dict['text'] = pre_texts
    return result_dict


if __name__ == '__main__':
    # loadnn = model_cls.Bert_Blend_CNN()
    sentences_tests_ = ['浙 江 省 城 市 水 业 协 会  江 苏 省 城 镇 供 水 排 水 协 会  安 徽 省 城 镇 供 水 协 会  上 海 市 供 水 行 业 协 会2019 （第二届）长三角三省一市一体化城镇供水合作发展论坛“寻找最美水务人”摄影大赛通知各供水企事业单位：为了迎接中华人民共和国成立 70 周年，推进水行业水务系统作风行风建设，弘扬水务人敬业奉献、敢于担当、不畏艰难、顽强拼搏的优秀品质，以身边先进事迹影响人、感召人、鼓舞人，进一步营造创先争优良好氛围。经长三角三省一市水协秘书长联席会议商定： 在 2019（第二届）长三角三省一市一体化城镇供水合作发展论坛之际，举办“寻找最美水务人”摄影大赛，展示城镇供水行业工作人员不畏严寒酷暑的优秀品质。现将本次“寻找最美水务人”摄影大赛的相关事宜告知如下：一、 主题定位“寻找最美水务人”摄影大赛是一次公益性的宣传活动，其目的是展示长三角三省 一市城镇供水一线工作人员敬业、爱岗精神。二、 大赛组织主办单位：浙江省城市水业协会江苏省城镇供水排水协会安徽省城镇供水协会上海市供水行业协会承办单位：上海展业展览有限公司三、 报名时间1 、 2019 年 1 月 20 日“寻找最美水务人”摄影大赛正式开始报名， 并上传参赛摄影作品，初赛截稿日期为 2019 年 2 月 20 日。2、 2 月 25 日起开始网上评选投票， 3 月 5 日网络评选投票截止。3、参赛选手关注大会官方微信公众号“水业圈”，点击下方“长三角会” —“大赛报名“，即可快速上传作品（强烈推荐）。如有疑问可将摄影作品直接发送至大会官方指定邮箱：2853266063@qq.Com，邮件主题请备注： 2019 长三角摄影大赛。四、 作品要求1、摄影作品内容符合“最美水务人”主题的均可参赛；2、本次比赛不限单幅和组照，传统照片和数码照片均可参赛；但不可对作品进行后期处理。3、图片格式要求： JPG,其他文件格式请转成本格式提交。图片像素不限，图片大小要求 2M --5M 以内，所有图片请保留 EXIF 信息。4、所有报送作品需附作品标题、简要说明（包括拍摄地点、时间、作品特点、背景故事等）、作者姓名、单位、联系电话、通讯地址、电子邮箱等。提交多幅作品的需自定编号（最多 4 张）。五、 奖项设置本次中影大赛采用网上公开投票及大会组委会专家投票评选方式， 3 月中旬公布摄影大赛结果， 颁奖典礼初定为 3 月 27 日。本次摄影大赛设置奖项为：特等奖一名 奖励 3000 元并颁发证书一等奖二名 奖励 2000 元并颁发证书二等奖三名 奖励 1000 元并颁发证书三等奖五名 奖励 500 元并颁发证书鼓励奖十名 奖励 200 元并颁发证书六、 免责声明所有参赛作品的著作权、与作品有关的肖像权和名誉权等法律问题，请参赛者自行解决并承担相应的责任。参赛者在提交作品时, 即默认作品版权为参赛者本人所有，并允许主办方在本次活动的相关专题、新闻、推广中无偿使用这些作品。谢绝提供电脑创意和改变原始影像的作品（照片仅可做亮度、对比度、色彩饱和度的适度调整， 不得做合成、添加、大幅度改变色彩等技术处理）。大赛主办方有权利要求参赛者提供图片原始文件。参赛者须保留原始正片、负片或数码作品原始文件, 以便调取获奖作品参加决选、印刷发表。如发现违规行为,则取消该参赛者的评奖资格。违反国家法律、法规的作品,或是主办方认为有违反公共秩序、社会风气的作品,将被取消参赛资格。奖品不得退换或者兑换现金，本次活动的最终解释权归主办单位,参加本次活动即视为同意并遵守本次活动各项规程。凡不符合规程的参赛者和作品将不具备参赛资格。七、 会务组联系方式陈 柳宋献英0571-87886551138 1869 0676邮箱： 406371828@qq.com邮箱： 2853266061@qq.com二零一九年 一 月十八日']
    result_dict = text_classification(sentences_tests_)
    print(result_dict)

    # model = torch.load('w')
    # model()

