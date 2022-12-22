base_dir = '/media/asura/T7_Shield_1/for_mid/20221216';
save_dir = '/media/asura/T7_Shield_1/for_mid/20221216/for_training';
subfolder_list = {'train', 'val', 'test'};
target_divide_folder_list = {'img', 'lidar', 'lidar_projected', 'mask', 'gray_mask'};
target_divide_folder_ext = {'jpg', 'pcd', 'png', 'png', 'png'};
subfolder_ratio = [0.8, 0.1, 0.1];

%% Divide randomly, assume that all file names are same.
sample_path = sprintf('%s/%s', base_dir, target_divide_folder_list{1});
sample_fileList = dir(fullfile(sample_path,'*.jpg'));
[trainInd,valInd,testInd] = dividerand(length(sample_fileList), subfolder_ratio(1), subfolder_ratio(2), subfolder_ratio(3));

mkdir(save_dir)
mkdir(sprintf('%s/%s', save_dir, 'train'))
mkdir(sprintf('%s/%s', save_dir, 'val'))
mkdir(sprintf('%s/%s', save_dir, 'test'))

for j=1:length(target_divide_folder_list)

    mkdir(sprintf('%s/%s/%s', save_dir, 'train', target_divide_folder_list{j}))
    mkdir(sprintf('%s/%s/%s', save_dir, 'val', target_divide_folder_list{j}))
    mkdir(sprintf('%s/%s/%s', save_dir, 'test', target_divide_folder_list{j}))

    source_path = sprintf('%s/%s', base_dir, target_divide_folder_list{j});
    source_fileList = dir(fullfile(source_path, sprintf('*.%s', target_divide_folder_ext{j})));

    parfor k=1:length(trainInd)
        target_path = sprintf('%s/%s/%s', save_dir, 'train', target_divide_folder_list{j});
        source_filename = source_fileList(trainInd(1, k)).name;
        source_file_path = sprintf('%s/%s', source_path, source_filename);
        target_file_path = sprintf('%s/%s', target_path, sprintf('%s.%s', source_filename(1:end-4), target_divide_folder_ext{j}));
        copyfile(source_file_path, target_file_path);
    end
    parfor k=1:length(valInd)
        target_path = sprintf('%s/%s/%s', save_dir, 'val', target_divide_folder_list{j});
        source_filename = source_fileList(valInd(1, k)).name;
        source_file_path = sprintf('%s/%s', source_path, source_filename);
        target_file_path = sprintf('%s/%s', target_path, sprintf('%s.%s', source_filename(1:end-4), target_divide_folder_ext{j}));
        copyfile(source_file_path, target_file_path);
    end
    parfor k=1:length(testInd)
        target_path = sprintf('%s/%s/%s', save_dir, 'test', target_divide_folder_list{j});
        source_filename = source_fileList(testInd(1, k)).name;
        source_file_path = sprintf('%s/%s', source_path, source_filename);
        target_file_path = sprintf('%s/%s', target_path, sprintf('%s.%s', source_filename(1:end-4), target_divide_folder_ext{j}));
        copyfile(source_file_path, target_file_path);
    end

end
