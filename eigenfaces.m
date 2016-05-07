% ---------------------------------------
% - Eigenfaces algorithm implementation -
% ---------- Valerian Saliou ------------
% ---------------------------------------

function eigenfaces()
    % First, train the database
    [database_sets, database_set_images, database_eigenfaces, database_mean_face, database_weights] = eigenfaces__train();
    
    % Then, use the database to classify faces
    eigenfaces__recognize(database_sets, database_set_images, database_eigenfaces, database_mean_face, database_weights);
end

function [database_sets, database_set_images, database_eigenfaces, database_mean_face, database_weights]=eigenfaces__train()
    disp('> Training started...');
    tic();
    
    [sets, set_images, images, image_height, image_width, image_count] = eigenfaces__load_images('training_set', 'pgm');
    
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
    eigenvectors = eigenfaces__process_eigenvectors(covariance_matrix);
    
    % TODO: use eigenvalues returned by <eigenfaces__process_eigenvectors> to
    % trim down to eigenvectors for highest eigenvalues (IF USING A LARGE
    % TRAINING SET)
    
    % Eigenfaces
    eigenfaces = eigenfaces__process_eigenfaces(eigenvectors, images_phi, image_height, image_width, image_count);
    
    % Weights
    weights = eigenfaces__process_weights(eigenfaces, images_phi, image_count);

    %eigenfaces__util_images_show(eigenfaces, image_height, image_width, image_count);
    
    fprintf('Processing time: %f seconds\n', toc());
    disp('> Training ended.');
    
    database_sets = sets;
    database_set_images = set_images;
    database_eigenfaces = eigenfaces;
    database_mean_face = image_psi;
    database_weights = weights;
end

function eigenfaces__recognize(database_sets, database_set_images, database_eigenfaces, database_mean_face, database_weights)
    disp('> Recognition started...');
    tic();
    
    [sets, set_images, images, image_height, image_width, image_count] = eigenfaces__load_images('recognition_set', 'pgm');
    
    disp(sprintf('Loaded %i images of %ix%i pixels', image_count, image_width, image_height));
    
    % Iterate on every image in the recognition set
    for i = 1:image_count
        image = images(((i - 1) * image_height + 1):(i * image_height), :);
        
        % Normalized face vectors: GAMMA(i){n} [GAMMA(1), GAMMA(2), ...]
        image_gamma = eigenfaces__normalize(image, image_height, image_width, 1);

        % Substracted mean face vectors: PHI(i) [PHI(1), PHI(2), ...]
        image_phi = eigenfaces__mean_substract(image_gamma, database_mean_face, image_height, image_width, 1);

        %eigenfaces__util_images_show(image_phi, image_height, image_width, 1);
        
        % Weights
        weights = eigenfaces__process_weights(database_eigenfaces, image_phi, 1);

        % Distances
        distances = eigenfaces__process_distances(weights, database_weights);
        
        [closest_weight, closest_index] = min(distances);
        farthest_weight = max(distances);
        
        if eigenfaces__process_is_match(closest_weight, distances) == true
            fprintf('HIT: %s/%s recognized as subject in set %s/%s\n', sets{i}, set_images{i}, database_sets{closest_index}, database_set_images{closest_index});
        elseif eigenfaces__process_is_face(closest_weight, distances) == true
            fprintf('MISS: %s/%s not found in any set\n', sets{i}, set_images{i});
        else
            fprintf('ERROR: %s/%s may not be an human face\n', sets{i}, set_images{i});
        end
        
        fprintf('Got weights: closest=%i; farthest=%i\n', closest_weight, farthest_weight);
    end
    
    fprintf('Processing time: %f seconds\n', toc());
    disp('> Recognition ended.');
end

function [sets, set_images, images, image_height, image_width, image_count]=eigenfaces__load_images(image_set, image_extension)
    sets = cell(0, 1);
    set_images = cell(0, 1);
    images = [];
    image_height = 0;
    image_width = 0;
    image_count = 0;
    
    % List classes
    directory_name = sprintf('/Users/valerian/Documents/ENSSAT/Imaging/Face Recognition/eigenfaces/%s/active', image_set);
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
            
            if image_extension ~= 'pgm'
                current_image = rgb2gray(current_image);
            end
            
            images = cat(1, images, current_image);
            
            sets{image_count} = [class_name];
            set_images{image_count} = [image_name];

            if i == 1
                [image_height, image_width] = size(current_image);
            end
        end
    end
end

function images_gamma=eigenfaces__normalize(images, image_height, image_width, image_count)
    images = eigenfaces__normalize_resize(images);
    images = eigenfaces__normalize_vector_project(images, image_height, image_width, image_count);
    images = eigenfaces__normalize_color_adjust(images);
    images = eigenfaces__normalize_position_adjust(images);
    
    images_gamma = images;
end

function images=eigenfaces__normalize_resize(images)
    % TODO
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

function images=eigenfaces__normalize_color_adjust(images)
    % TODO
end

function images=eigenfaces__normalize_position_adjust(images)
    % TODO
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
    covariance_matrix = images_phi * images_phi';
end

function [eigenvectors, eigenvalues]=eigenfaces__process_eigenvectors(covariance_matrix)
    [eigenvectors, eigenvalues] = eig(covariance_matrix);
    
    eigenvectors = fliplr(eigenvectors);
    
    eigenvalues = diag(eigenvalues);
    eigenvalues = eigenvalues(end:-1:1);
end

function eigenfaces=eigenfaces__process_eigenfaces(eigenvectors, images_phi, image_height, image_width, image_count)
    % Multiply {ith} eigenvectors by the whole normalized image set
    % This gives the {ith} eigenface
    for i = 1:image_count
        eigenfaces(i, :) = eigenvectors(:, i)' * images_phi;
    end
    
    % Normalize pixels from 0 to 255
    pixel_min = min(min(eigenfaces));
    pixel_max = max(max(eigenfaces));
    
    for i = 1:image_count
        for p = 1:(image_height * image_width)
            eigenfaces(i, p) = 255 * (eigenfaces(i, p) - pixel_min) / (pixel_max - pixel_min);
        end
    end
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

function is_match=eigenfaces__process_is_match(distance, distances)
    is_match = false;
    
    % TODO: dynamic error / threshold calculation
    % @ref: http://matlabsproj.blogspot.com/2012/06/face-recognition-using-eigenfaces_11.html
    % @see: Thresholds for Eigenface Recognition
    
    threshold = 1;  % This one is set empirically (FIXME)
    
    if distance < threshold
        is_match = true;
    end
end

function is_face=eigenfaces__process_is_face(distance, distances)
    is_face = false;
    
    % TODO: dynamic error / threshold calculation
    % @ref: http://matlabsproj.blogspot.com/2012/06/face-recognition-using-eigenfaces_11.html
    % @see: Thresholds for Eigenface Recognition
    
    threshold_factor = 1 / 4;  % This one is set empirically (FIXME)
    threshold = threshold_factor * max(distances);
    
    if distance < threshold
        is_face = true;
    end
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