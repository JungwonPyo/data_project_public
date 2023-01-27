base_path = '/media/asura/Dataset/dataset/parking_dataset/total_for_training/for_training';

sub_lists = {'train', 'val', 'test'};

for n=1:size(sub_lists, 2)

    sub_filename = sub_lists{n};
%     img_foldername = 'img';
    img_foldername = 'de_front';
%     label_foldername = 'gray_mask';
%     label_foldername = 'mask';
    label_foldername = 'mask_front';
    listname = sprintf('%s_list.lst', sub_filename);
    
    full_img_path = sprintf('%s/%s/%s', base_path, sub_filename, img_foldername);
    full_label_path = sprintf('%s/%s/%s', base_path, sub_filename, label_foldername);
    fileList = dir(fullfile(full_img_path,'*.jpg'));
    fid = fopen(sprintf('%s/%s/%s', base_path, sub_filename, listname),'w');
    for i = 1:length(fileList)
        img_filename = sprintf('%s/%s', full_img_path, fileList(i).name);
        temp_label_name = sprintf('%s.png', fileList(i).name(1:end-4));
        label_filename = sprintf('%s/%s', full_label_path, temp_label_name);
        if isfile(label_filename)
            full_sent = sprintf('%s\t%s\n', img_filename, label_filename);
            fprintf(fid, '%s', full_sent);
        else
            warningMessage = sprintf('Warning: file does not exist:\n%s', fullFileName);
            break
        end
    end
    fclose(fid);

end
