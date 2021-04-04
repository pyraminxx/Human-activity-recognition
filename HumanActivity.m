clear all, close all; clc

 
trn_set = imageDatastore('train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames',...
    'FileExtensions', { '.jpg'}); 

tst_set = imageDatastore('test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames',...
    'FileExtensions', { '.jpg'}); 


%%
% netstore : googlenet inceptionv3 mobilenetv2 resnet18 resnet50 xception  
net = googlenet;
inputSize = net.Layers(1).InputSize


%%
trn_set.ReadFcn = @read_img;
tst_set.ReadFcn = @read_img;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 



numClasses = numel(categories(trn_set.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Name','Layers','NumberTitle','off');
% plot(lgraph)


%%

layers = lgraph.Layers;
connections = lgraph.Connections;

Nfreeze = 20;
layers(1:end-Nfreeze) = freezeWeights(layers(1:end-Nfreeze));
lgraph = createLgraphUsingConnections(layers,connections);


%%
% pixelRange = [-30 30];
% scaleRange = [0.9 1.1];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);
% 
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),trn_set, ...
%     'DataAugmentation',imageAugmenter);

% minibatch = preview(augimdsTrain);
% imshow(imtile(minibatch.input));
 
%%
reset(gpuDevice)
% sgdm  rmsprop adam
miniBatchSize = 32;
valFrequency = 2*floor(numel(trn_set.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',2,...
    'LearnRateDropFactor',0.75,...
    'Shuffle','every-epoch', ...
    'ValidationData',tst_set, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');

tic
[net, net_stat]= trainNetwork(trn_set,lgraph,options);
cnn_train_time = toc
%%
save('googlenet_sgd_transfer_20.mat','net_stat')

%%
tic
[YPred,probs] = classify(net,tst_set);
cnn_test_time = toc

C = confusionmat( tst_set.Labels  ,YPred) ;

fc=figure('Name','Heatmap','NumberTitle','off','Position', get(0, 'Screensize'))
%xvalues={'applauding','blowing_bubbles','brushing_teeth','cleaning_the_floor','climbing','cooking','cutting_trees','cutting_vegetables','drinking','feeding_a_horse','fishing','fixing_a_bike','fixing_a_car','gardening','holding_an_umbrella','jumping','looking_through_a_microscope','looking_through_a_telescope','playing_guitar','playing_violin','pouring_liquid','pushing_a_cart','reading','phoning','riding_a_bike','riding_a_horse','rowing_a_boat','running','shooting_an_arrow','smoking','taking_photos','texting_message','throwing_frisby','using_a_computer','walking_the_dog','washing_dishes','watching_TV','waving_hands','writing_on_a_board','writing_on_a_book'};
%yvalues={'applauding','blowing_bubbles','brushing_teeth','cleaning_the_floor','climbing','cooking','cutting_trees','cutting_vegetables','drinking','feeding_a_horse','fishing','fixing_a_bike','fixing_a_car','gardening','holding_an_umbrella','jumping','looking_through_a_microscope','looking_through_a_telescope','playing_guitar','playing_violin','pouring_liquid','pushing_a_cart','reading','phoning','riding_a_bike','riding_a_horse','rowing_a_boat','running','shooting_an_arrow','smoking','taking_photos','texting_message','throwing_frisby','using_a_computer','walking_the_dog','washing_dishes','watching_TV','waving_hands','writing_on_a_board','writing_on_a_book'};

h=heatmap(round(100*bsxfun(@rdivide,C,sum(C,2))))
h.Title='Confusion Matrix';
h.XLabel='Predictions';
h.YLabel='GroundTruth';

[acc,pre,rcl,fss,kp] = statConf(C);

disp('Accuracy etc.')
disp(100*[acc,pre,rcl,fss])
%%
saveas(fc,'googlenet_adam_confusion_transfer_20.png')
%%

trn_mat = activations(net , trn_set, lgraph.Layers(end-3).Name , 'OutputAs' , 'rows');
tst_mat = activations(net , tst_set, lgraph.Layers(end-3).Name , 'OutputAs' , 'rows');

trn_gnd = double(trn_set.Labels);
tst_gnd = double(tst_set.Labels);
%%


tic
t = templateSVM('Standardize',false,...
                'KernelFunction','linear',...
                'KernelScale','auto' );
                
Mdl = fitcecoc(trn_mat,trn_gnd,'Learners',t);
svm_train_time = toc

tic
pre_scr_svm = predict(Mdl ,tst_mat );
svm_test_time = toc

C_svm = confusionmat( tst_gnd ,pre_scr_svm) ;
[acc,pre,rcl,fss,kp] = statConf(C_svm);
disp('SVM result')
disp(100*[acc,pre,rcl,fss])

%%
% inverse squaredinverse
tic
Mdl = fitcknn(trn_mat,trn_gnd,...
    'NumNeighbors',7,...
    'DistanceWeight', 'equal' );
knn_train_time = toc

tic
pre_scr_svm = predict(Mdl ,tst_mat );
knn_test_time = toc

C_knn = confusionmat( tst_gnd ,pre_scr_svm) ;
[acc,pre,rcl,fss,kp] = statConf(C_knn);

disp('KNN result')
disp(100*[acc,pre,rcl,fss])
%%

ydata = tsne(trn_mat, trn_gnd, 2 );

class_mean = [];
for i = 1:40
class_mean(i,:) = mean(ydata(trn_gnd==i,:),1);
end

figure('Name','TSNE_train','NumberTitle','off')
scatter(ydata(:,1),ydata(:,2),9,trn_gnd,'filled') , hold on
scatter(class_mean(:,1),class_mean(:,2),15,'d','filled')
b = num2str([1:40]'); 
c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(class_mean(:,1)+dx, class_mean(:,2)+dy, c,'FontSize',14,'FontWeight','bold');

%%

ydata = tsne(tst_mat, tst_gnd, 2 );


class_mean = [];
for i = 1:40
class_mean(i,:) = mean(ydata(trn_gnd==i,:),1);
end

ft=figure('Name','TSNE_test','NumberTitle','off')
scatter(ydata(:,1),ydata(:,2),9,tst_gnd,'filled') , hold on
scatter(class_mean(:,1),class_mean(:,2),15,'d','filled')
b = num2str([1:40]'); 
c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(class_mean(:,1)+dx, class_mean(:,2)+dy, c,'FontSize',14,'FontWeight','bold');
%%
saveas(ft,'inceptionv3_tsne.png')
%%



