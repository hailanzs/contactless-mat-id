clear;
clc;

% TODO: Replace with the calibration matrix for XWR1843BOOST
cm  = [0,	-8.11, 14.52,	6.98, 10.92, 4.4, 42.06,	35.28];
cm = reshape(cm, [2,4]);
cm = cm .* (pi / 180);

for day = [""]
for exp_num = 0:1000

        folder = pwd;
        orig_path = sprintf("%s/measured_data/%s", folder, day)
        calib_path = sprintf("%s/calibrated_data/%s", folder, day);
        if ~exist(calib_path, 'dir')
            mkdir(calib_path)
        end
        filepath = fullfile(orig_path, string(exp_num));
        try
            load(filepath);
            [num_frms, tot] = size(raw_data);
            rx = 4;
            tx = 2;
            adc_sample = 64;
            reorder_raw = zeros([num_frms,tot/2]);
            result = zeros([num_frms,tx,rx,adc_sample]);
%           Separate IQ data
            for ii = 1:num_frms
                reorder_raw(ii,1:2:end) = raw_data(ii,1:4:end) + 1j * raw_data(ii,3:4:end);
                reorder_raw(ii,2:2:end) = raw_data(ii,2:4:end) + 1j * raw_data(ii,4:4:end);
            end
            reorder_raw = reshape(reorder_raw, [num_frms, adc_sample,  rx, tx]);
            reorder_raw = permute(reorder_raw, [1,4,3,2]);
            calib_matrix = zeros(2,4);
            for i = 1:length(reorder_raw)
                for h = 1:length(reorder_raw(1,1,1,:))
                    calib_matrix = exp(-1j*cm);
                    calibrated_result = calib_matrix .* squeeze(reorder_raw(i,:,:,h));
                    result(i,:,:,h) = calibrated_result;
                end
            end
            save(sprintf('%s\%s.mat',calib_path, int2str(exp_num)), 'sample_rate', 'details','date', 'exp_num', 'exp_object',  'result');
            "done with " + string(exp_num) + " " + string(day)
        catch
            continue
        end
    end
end
