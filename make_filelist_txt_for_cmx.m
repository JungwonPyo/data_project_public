base_path = '/media/kkk/T7_Shield';
target_path = '/media/kkk/T7_Shield/final_for_training';
train_foldername = 'train';
train_listname = 'train.txt';
val_foldername = 'val';
val_listname = 'val.txt';
test_foldername = 'test';
test_listname = 'test.txt';
% img_foldername = 'img';
img_foldername = 'de_front';

D = sprintf('%s/%s/%s', target_path, train_foldername, img_foldername);
fileList = dir(fullfile(D,'*.jpg'));
fid = fopen(sprintf('%s/%s', base_path, train_listname),'w');
for i = 1:length(fileList)
  fprintf(fid,'%s\n',fileList(i).name(1:end-4));
end
fclose(fid);

D = sprintf('%s/%s/%s', target_path, val_foldername, img_foldername);
fileList = dir(fullfile(D,'*.jpg'));
fid = fopen(sprintf('%s/%s', base_path, val_listname),'w');
for i = 1:length(fileList)
  fprintf(fid,'%s\n',fileList(i).name(1:end-4));
end
fclose(fid);

D = sprintf('%s/%s/%s', target_path, test_foldername, img_foldername);
fileList = dir(fullfile(D,'*.jpg'));
fid = fopen(sprintf('%s/%s', base_path, test_listname),'w');
for i = 1:length(fileList)
  fprintf(fid,'%s\n',fileList(i).name(1:end-4));
end
fclose(fid);
