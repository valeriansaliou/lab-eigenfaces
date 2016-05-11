% ---------------------------------------
% - Eigenfaces algorithm implementation -
% ---------- Valerian Saliou ------------
% ---------------------------------------

function eigenfaces()
    clear;
    
    % Maximum size (images will be resized if larger)
    image_max_width = 46;
    image_max_height = 56;
    
    % First, train the database
    [database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights] = eigenfaces__train(image_max_width, image_max_height);
    
    % Then, use the database to classify faces
    eigenfaces__recognize(image_max_width, image_max_height, database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights);
    
    % Process validation
    eigenfaces__validation(database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights);
end

function [database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights]=eigenfaces__train(image_max_width, image_max_height)
    disp('> Training started...');
    tic();
    
    [sets, set_images, images, image_height, image_width, image_count] = eigenfaces__load_images('training_set', image_max_width, image_max_height);
    
    disp(sprintf('Loaded %i images of %ix%i pixels', image_count, image_width, image_height));
    
    % Normalized face vectors: GAMMA(i){n} [GAMMA(1), GAMMA(2), ...]
    images_gamma = eigenfaces__normalize(images, image_height, image_width, image_count);
    
    % Mean face vector: PSI{1}
    image_psi = eigenfaces__mean(images_gamma, image_height, image_width, image_count);
    
    % Substracted mean face vectors: PHI(i) [PHI(1), PHI(2), ...]
    images_phi = eigenfaces__mean_substract(images_gamma, image_psi, image_height, image_width, image_count);
    
    % Covariance matrix
    covariance_matrix = eigenfaces__process_covariance_matrix(images_phi);
    
    % Eigenvectors
    eigenvectors = eigenfaces__process_eigenvectors(covariance_matrix, image_count);
    
    % Eigenfaces
    eigenfaces = eigenfaces__process_eigenfaces(eigenvectors);
   
    % Weights
    weights = eigenfaces__process_weights(eigenfaces, images_phi, image_count);
    
    % Uncomment this to view PHI images
    %eigenfaces__util_images_show(eigenfaces(1:10, :), image_height, image_width, size(eigenfaces(1:10, :), 1));
    
    fprintf('Processing time: %f seconds\n', toc());
    disp('> Training ended.');
    
    database_sets = sets;
    database_set_images = set_images;
    database_images = images;
    database_eigenfaces = eigenfaces;
    database_mean_face = image_psi;
    database_weights = weights;
end

function eigenfaces__recognize(image_max_width, image_max_height, database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights)
    disp('> Recognition started...');
    tic();
    
    [sets, set_images, images, image_height, image_width, image_count] = eigenfaces__load_images('recognition_set', image_max_width, image_max_height);
    
    disp(sprintf('Loaded %i images of %ix%i pixels', image_count, image_width, image_height));
    
    distances_all = [];
    
    % Iterate on every image in the recognition set (build distances)
    for i = 1:image_count
        image = eigenfaces__util_image_from_vector(images, image_height, image_width, i);
        
        % Normalized face vectors: GAMMA(i){n} [GAMMA(1), GAMMA(2), ...]
        image_gamma = eigenfaces__normalize(image, image_height, image_width, 1);

        % Substracted mean face vectors: PHI(i) [PHI(1), PHI(2), ...]
        image_phi = eigenfaces__mean_substract(image_gamma, database_mean_face, image_height, image_width, 1);

        % Uncomment this to view PHI images
        %eigenfaces__util_images_show(image_phi, image_height, image_width, 1);
        
        % Weights
        weights = eigenfaces__process_weights(database_eigenfaces, image_phi, 1);

        % Distances
        distances = eigenfaces__process_distances(weights, database_weights);
        
        distances_all(i, :) = distances;
    end
    
    % Vectorize minimum distances
    minimum_distances = min(distances_all');
    
    % Take a decision (build results)
    results_all = [];
    
    for i = 1:image_count
        distances = distances_all(i, :);
        
        [closest_weight, closest_index] = min(distances);
        farthest_weight = max(distances);
        
        [is_face, is_match]=eigenfaces__process_matcher(closest_weight, minimum_distances);
        
        if is_match == true
            result = closest_index;
            
            fprintf('HIT: %s/%s recognized as subject in set %s/%s\n', sets{i}, set_images{i}, database_sets{closest_index}, database_set_images{closest_index});
            
        elseif is_face == true
            result = 0;
            
            fprintf('MISS: %s/%s not found in any set\n', sets{i}, set_images{i});
        else
            result = -1;
            
            fprintf('ERROR: %s/%s may not be an human face\n', sets{i}, set_images{i});
        end
        
        results_all(i) = result;
        
        fprintf('Got weights: closest=%i; farthest=%i\n', closest_weight, farthest_weight);
    end
    
    eigenfaces__util_recognition_show(images, database_images, sets, database_sets, results_all, image_height, image_width, image_count);
    
    fprintf('Processing time: %f seconds\n', toc());
    disp('> Recognition ended.');
end

function eigenfaces__validation(image_max_width, image_max_height, database_sets, database_set_images, database_images, database_eigenfaces, database_mean_face, database_weights)
    disp('> Validation started...');
    tic();
    
    % TODO: validate the quality of implementation
    %  -> error rate
    %  -> speed per recognition unit
    
    fprintf('Processing time: %f seconds\n', toc());
    disp('> Validation ended.');
end

function [sets, set_images, images, image_height, image_width, image_count]=eigenfaces__load_images(image_set, image_max_width, image_max_height)
    sets = cell(0, 1);
    set_images = cell(0, 1);
    images = [];
    image_height = 0;
    image_width = 0;
    image_count = 0;
    
    image_extension = 'pgm';
    
    % List classes
    directory_name = sprintf('./%s/active', image_set);
    class_dirs = dir(directory_name);
    class_index = find([class_dirs.isdir]);

    for c = 1:length(class_index)
        class_name = class_dirs(class_index(c)).name;
        class_path = fullfile(directory_name, class_name);
        
        % List images
        image_files = dir(fullfile(class_path, sprintf('*.%s', image_extension)));
        image_index = find(~[image_files.isdir]);
        
        for i = 1:length(image_index)
            image_count = image_count + 1;
            
            image_name = image_files(image_index(i)).name;
            image_path = fullfile(class_path, image_name);

            current_image = imread(image_path);
            
            % Convert to grayscale?
            if image_extension ~= 'pgm'
                current_image = rgb2gray(current_image);
            end
            
            % Resize? (if Image Processing Toolbox is available)
            if size(current_image, 1) > image_max_height || size(current_image, 2) > image_max_width
                current_image = eigenfaces__util_image_resize(current_image, image_max_width, image_max_height);
            end
            
            images = cat(1, images, current_image);
            
            sets{image_count} = [class_name];
            set_images{image_count} = [image_name];
            
            [current_image_height, current_image_width] = size(current_image);

            if i == 1
                % First image size is reference size
                image_height = current_image_height;
                image_width = current_image_width;
            else
                % Check next images size matches that of first image (all
                % images MUST have the sa~me size)
                if current_image_height ~= image_height || current_image_width ~= image_width
                    throw(MException('MYFUN:image_size', 'Images must all have the same size'));
                end
            end
        end
    end
end

function images_gamma=eigenfaces__normalize(images, image_height, image_width, image_count)
    images = eigenfaces__normalize_vector_project(images, image_height, image_width, image_count);

    images_gamma = images;
end

function images_vector=eigenfaces__normalize_vector_project(images, image_height, image_width, image_count)
    images_vector = zeros(image_count, image_height * image_width);
    
    for i = 1:image_count
        image_vector = images(((i - 1) * image_height + 1):(i * image_height), :);
        image_vector = reshape(image_vector', image_height * image_width, 1)';
        image_vector = double(image_vector);
        
        images_vector(i, :) = image_vector;
    end
end

function image_psi=eigenfaces__mean(images, image_height, image_width, image_count)
    image_psi = zeros(1, image_width * image_height);
    
    for i = 1:image_count
        for c = 1:(image_width * image_height)
            image_psi(1, c) = image_psi(1, c) + (images(i, c) / image_count);
        end
    end
end

function images_phi=eigenfaces__mean_substract(images_gamma, image_psi, image_height, image_width, image_count)
    images_phi = zeros(image_count, image_height * image_width);
    
    for i = 1:image_count
        images_phi(i, :) = images_gamma(i, :) - image_psi;
    end
end

function covariance_matrix=eigenfaces__process_covariance_matrix(images_phi)
    covariance_matrix = cov(images_phi);
end

function [eigenvectors, eigenvalues]=eigenfaces__process_eigenvectors(covariance_matrix, image_count)
    [eigenvectors, eigenvalues] = eig(covariance_matrix);
    
    % Adjust eigenvectors + eigenvalues working sets
    eigenvectors = fliplr(eigenvectors);
    eigenvectors = eigenvectors';
    
    eigenvalues = diag(eigenvalues);
    eigenvalues = eigenvalues(end:-1:1);
    
    % Pick 15% largest eigenvalues, and split eigenvectors accordingly
    % @see: http://globaljournals.org/GJCST_Volume10/gjcst_vol10_issue_1_paper10.pdf
    % @ref: Page 1
    split_factor = 0.15;
    split_limit = ceil(size(eigenvectors, 1) * split_factor);
    
    if split_limit < 1
        split_limit = 1;
    end
    
    eigenvectors = eigenvectors(1:split_limit, :);
    eigenvalues = eigenvalues(1:split_limit, 1);
end

function eigenfaces=eigenfaces__process_eigenfaces(eigenvectors)
    % Eigenfaces is another word for eigenvectors
    eigenfaces = eigenvectors;
end

function weights=eigenfaces__process_weights(eigenfaces, images_phi, image_count)
    weights = [];
    
    for i = 1:size(eigenfaces, 1)
        for j = 1:image_count
            weights(i, j) = dot(images_phi(j, :), eigenfaces(i, :));
        end
    end
end

function distances=eigenfaces__process_distances(weights, database_weigths)
    distances = [];
    
    for i = 1:size(weights, 2)
        for j = 1:size(database_weigths, 2)
            distances(i, j) = norm(weights(:, i) - database_weigths(:, j));
        end
    end
end

function [is_face, is_match]=eigenfaces__process_matcher(distance, minimum_distances)
    % Dynamic threshold processing, based on maximum value of the set of
    % minimum distances
    % Ignore values above a certain 'non-face' threshold: this avoids non-human
    % objects to mess-up the recognition algorithm results
    % @see: http://globaljournals.org/GJCST_Volume10/gjcst_vol10_issue_1_paper10.pdf
    % @ref: Page 3
    
    is_face = false;
    is_match = false;
    
    non_face_threshold = 5.0e003;
    
    if distance < non_face_threshold
        is_face = true;
        
        non_match_threshold = 0.8 * max(minimum_distances(minimum_distances < non_face_threshold));

        if distance < non_match_threshold
            is_match = true;
        end
    end
end

function eigenfaces__util_recognition_show(images, database_images, sets, database_sets, results_all, image_height, image_width, image_count)
    figure;
    
    plot_grid_width = 2;
    plot_grid_height = image_count;
    
    for i = 1:image_count
        image = eigenfaces__util_image_from_vector(images, image_height, image_width, i);
        
        % Show input image {i}
        subplot(plot_grid_height, plot_grid_width, 2 * i - 1);
        imshow(image, 'DisplayRange', [0 255]);
        title(sprintf('Input is %s', sets{i}));
        
        result_image_compare = zeros(image_height, image_width);
        result_index = results_all(i);
        
        if result_index > 0
            result_image_compare = eigenfaces__util_image_from_vector(database_images, image_height, image_width, result_index);
        end
        
        subplot(plot_grid_height, plot_grid_width, 2 * i);
        imshow(result_image_compare, 'DisplayRange', [0 255]);
        
        if result_index > 0
            title(sprintf('Recognized as %s', database_sets{result_index}));
        elseif result_index == 0
            title('Not found');
        else
            title('Not a face');
        end
    end
end

function image=eigenfaces__util_image_from_vector(image_vector, image_height, image_width, image_index)
    image = image_vector(((image_index - 1) * image_height + 1):(image_index * image_height), :);
end

function image=eigenfaces__util_images_matrix_from_vector(image_vector, image_height, image_width)
    for y = 1:image_height
        for x = 1:image_width
            image(y, x) = image_vector(1, x + (y - 1) * image_width);
        end
    end
end

function eigenfaces__util_images_show(images, image_height, image_width, image_count)
    images_matrix = [];
    
    for i = 1:image_count
        current_image_matrix = eigenfaces__util_images_matrix_from_vector(images(i, :), image_height, image_width);
        
        images_matrix = cat(1, images_matrix, current_image_matrix);
    end
    
    imshow(images_matrix, 'DisplayRange', [0 255]);
end

function resized_image=eigenfaces__util_image_resize(image, target_width, target_height)
    % Modified snippet from: http://stackoverflow.com/questions/6183155/resizing-an-image-in-matlab
    % Prevents using imresize(), which requires Image Processing Toolbox

    scale_zoom = 1;
  
	if size(image, 1) > target_width
        scale_zoom = target_width / size(image, 1);
    end
    
	if size(image, 2) > target_height
        scale_zoom = target_height / size(image, 1);
    end

	old_size = size(image);
	new_size = max(floor(scale_zoom .* old_size(1:2)), 1);
    
	new_x = ((1:new_size(2)) - 0.5) ./ scale_zoom + 0.5;
	new_y = ((1:new_size(1)) - 0.5) ./ scale_zoom + 0.5;
    
	old_class = class(image);
    
	image = double(image);

	resized_image = interp2(image, new_x, new_y(:), 'cubic');
	resized_image = cast(resized_image, old_class);
end