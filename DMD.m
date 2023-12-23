
clc

% Load data
data = readtable("/Users/mahanivethakannappan/Downloads/your_modified_file.csv");

% Ask user to choose option
disp('Choose an option:');
disp('1. Enter date manually');
disp('2. Enter activity label');
disp('3. Enter specific time');
option = input('Enter your choice (1, 2, or 3): ');

if option == 1
    % Option 1: Enter date manually

    % Show available dates
    availableDates = unique(data.date);
    disp('Available dates:');
    disp(availableDates);

    % Ask for date input
    dateInputStr = input('Enter date (e.g., 24-03-2021 ): ', 's');

    % Convert input to datetime with the correct format
    dateInput = datetime(dateInputStr, 'InputFormat', 'dd-MM-yyyy');

   
    % Filter data for the given timestamp
    selectedData = data(ismember(data.date, dateInput), :);

elseif option == 2
    % Option 2: Enter activity label

    % Show available activity labels
    disp('Available activity labels:');
    disp(unique(data.label));

    % Ask for activity label input
    labelInput = input('Enter activity label: ');

    % Filter data for the selected label
    selectedData = data(data.label == labelInput, :);

    % Find the person who performed the activity the most
    uniquePersons = unique(selectedData.source);
    activityCounts = zeros(size(uniquePersons));

    for i = 1:length(uniquePersons)
        activityCounts(i) = sum(selectedData.source == uniquePersons(i));
    end

    [~, maxIdx] = max(activityCounts);
    mostActivePerson = uniquePersons(maxIdx);

    disp(['Person ', num2str(mostActivePerson), ' performed activity ', num2str(labelInput), ' the most.']);

    % Terminate the program after displaying the most active person
    return;

elseif option == 3
    % Option 3: Enter specific time

        % Convert 'time' column to string and remove seconds temporarily
    data.time = cellstr(data.time);
    data.time = cellfun(@(x) x(1:5), data.time, 'UniformOutput', false);

    % Ask for timestamp input
    timeInputStr = input('Enter time (e.g., 14:30): ', 's');

    % Extract only the hours and minutes from the input
    try
        timeInput = datetime(timeInputStr, 'InputFormat', 'HH:mm');
    catch
        error('Invalid time format. Please enter the time in the format HH:mm.');
    end

    % Create a new column for the entered timestamp
    data.enteredDate = datetime(data.time, 'InputFormat', 'HH:mm');

    % Filter data for the given entered timestamp
    selectedData = data(ismember(data.enteredDate, timeInput), :);


    % Set dateInput for plotting purposes
    dateInput = timeInput;

else
    disp('Invalid option. Please choose 1, 2, or 3.');
    return;
end

% Display available persons
disp('Available persons:');
disp(unique(selectedData.source));

% Ask for person number
personInput = input('Enter person number: ');

% Filter data for the selected person
personSubset = selectedData(selectedData.source == personInput, :);

% Get unique labels
labels = {'Walking', 'Shuffling', 'Stairs (Ascending)', 'Stairs (Descending)', 'Standing', 'Sitting', 'Lying'};

% Count occurrences of each label
labelCounts = histcounts(personSubset.label, 1:numel(labels) + 1);

% Bar plot
figure;
bar(labelCounts);
title(['Activities on ', datestr(dateInput)]);
xlabel('Label');
ylabel('Count');
xticks(1:numel(labels));
xticklabels(labels);

% Dynamic Mode Decomposition (DMD) for selected person's data
X = personSubset{:, [4:9]}; % Assuming columns 4 to 9 are the relevant data columns
dmd_results = dmd(X(:, 1:end-1), X(:, 2:end));

% Display DMD results
Phi = dmd_results.Phi;
Lambda = dmd_results.Lambda;

disp('DMD Results:');
disp('Phi:');
disp(Phi);
disp('Lambda:');
disp(Lambda);

% DMD Mode Frequencies
mode_frequencies = imag(log(diag(Lambda))) / (2 * pi);
figure;
stem(1:numel(mode_frequencies), mode_frequencies);
title('DMD Mode Frequencies');
xlabel('Mode');
ylabel('Frequency (Hz)');

% Reconstructed Data with Selected Modes
subset_modes = 1:3;
reconstructed_data = Phi(:, subset_modes) * diag(Lambda(subset_modes)) * pinv(Phi(:, subset_modes));
figure;
subplot(2, 1, 1);
plot(X(:, 1:end-1), 'b-', 'LineWidth', 1.5);
title('Original Data');
xlabel('Time');
ylabel('Amplitude');
subplot(2, 1, 2);
plot(reconstructed_data, 'r--', 'LineWidth', 1.5);
title('Reconstructed Data (Selected Modes)');
xlabel('Time');
ylabel('Amplitude');

% Mode Amplitude Evolution Over Time
mode_amplitudes = abs(Phi);
figure;
for i = 1:size(mode_amplitudes, 2)
    subplot(2, ceil(size(mode_amplitudes, 2)/2), i);
    plot(personSubset.date, mode_amplitudes(:, i), 'o-');
    title(['Mode ', num2str(i), ' Amplitude']);
    xlabel('Time');
    ylabel('Amplitude');
end

% Sensitivity Analysis
perturbed_data = X + randn(size(X)) * 0.1; % Adjust the perturbation level as needed
perturbed_dmd_results = dmd(perturbed_data(:, 1:end-1), perturbed_data(:, 2:end));

% Compare original and perturbed DMD modes
figure;
subplot(1, 2, 1);
plot(real(Lambda), imag(Lambda), 'o');
title('Original DMD Eigenvalues');

subplot(1, 2, 2);
plot(real(perturbed_dmd_results.Lambda), imag(perturbed_dmd_results.Lambda), 'o');
title('Perturbed DMD Eigenvalues');

% Statistical Analysis
mean_back = mean(personSubset{:, {'back_x', 'back_y', 'back_z'}});
std_back = std(personSubset{:, {'back_x', 'back_y', 'back_z'}});
range_back = range(personSubset{:, {'back_x', 'back_y', 'back_z'}});

mean_thigh = mean(personSubset{:, {'thigh_x', 'thigh_y', 'thigh_z'}});
std_thigh = std(personSubset{:, {'thigh_x', 'thigh_y', 'thigh_z'}});
range_thigh = range(personSubset{:, {'thigh_x', 'thigh_y', 'thigh_z'}});

disp('Back Sensor Statistics:');
disp(['Mean: ', num2str(mean_back)]);
disp(['Standard Deviation: ', num2str(std_back)]);
disp(['Range: ', num2str(range_back)]);

disp('Thigh Sensor Statistics:');
disp(['Mean: ', num2str(mean_thigh)]);
disp(['Standard Deviation: ', num2str(std_thigh)]);
disp(['Range: ', num2str(range_thigh)]);

% Correlation Analysis
correlation_matrix = corrcoef(personSubset{:, {'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'}});

figure;
heatmap({'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'}, {'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'}, correlation_matrix);
title('Correlation Matrix');

% Predictive Modeling
features = personSubset{:, {'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'}};
labels = personSubset.label;

% Split data into training and testing sets
rng(42); % Set seed for reproducibility
split_ratio = 0.8;
split_index = round(split_ratio * height(personSubset));
train_features = features(1:split_index, :);
train_labels = labels(1:split_index);
test_features = features(split_index+1:end, :);
test_labels = labels(split_index+1:end);

% Train a multi-class SVM model using fitcecoc
svmModel = fitcecoc(train_features, train_labels);

% Extract support vectors and their indices
supportVectorIndices = false(size(train_labels));

% Loop through binary SVM models and accumulate support vectors
for i = 1:numel(svmModel.BinaryLearners) 
    binarySVM = svmModel.BinaryLearners{i};
    predictedLabels = predict(binarySVM, train_features); %use current binary svn=m to predict
    supportVectorIndices = supportVectorIndices | (predictedLabels == train_labels); %mark sample as support vector
end

% Extract support vectors
supportVectors = train_features(supportVectorIndices, :);

% Plot the decision boundary and support vectors
figure;

% Scatter plot of training data
gscatter(train_features(:, 1), train_features(:, 2), train_labels, 'rgb', 'osd');
hold on;

% Highlight support vectors
plot(supportVectors(:, 1), supportVectors(:, 2), 'ko', 'MarkerSize', 10);

% Plot the decision boundary
minX = min(train_features(:, 1));
maxX = max(train_features(:, 1));
minY = min(train_features(:, 2));
maxY = max(train_features(:, 2));

[X, Y] = meshgrid(minX:(maxX-minX)/100:maxX, minY:(maxY-minY)/100:maxY);
Z = predict(svmModel, [X(:), Y(:), zeros(size(X(:), 1), 4)]); % Ensure the input has 6 columns

contour(X, Y, reshape(Z, size(X)), 'LineColor', 'k', 'LineWidth', 2);

hold off;

xlabel('Feature 3');
ylabel('Feature 4');
title('SVM Decision Boundary and Support Vectors');
legend('Class 1', 'Class 2', 'Class 3', 'Support Vectors', 'Decision Boundary', 'Location', 'Best');



% Evaluate the SVM model
svmPredictions = predict(svmModel, test_features); %Predict label for testfeatures
svmAccuracy = sum(svmPredictions == test_labels) / length(test_labels);
disp(['SVM Accuracy: ', num2str(svmAccuracy)]);

% Calculate accuracy for each class
numClasses = numel(unique(test_labels));
classAccuracies = zeros(1, numClasses);

for i = 1:numClasses
    classIndices = test_labels == i;
    classAccuracies(i) = sum(svmPredictions(classIndices) == test_labels(classIndices)) / sum(classIndices);
end

% Plot bar graph of accuracies for each class
figure;
bar(1:numClasses, classAccuracies);
xlabel('Class');
ylabel('Accuracy');
title('Accuracy for Each Class - SVM Model');

% Create a confusion matrix
confMat = confusionmat(test_labels, svmPredictions);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Create a confusion chart (heatmap)
figure;
confusionchart(test_labels, svmPredictions, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Confusion Matrix - SVM Model');


% Function for Dynamic Mode Decomposition (DMD)
function dmd_results = dmd(X1, X2)
    % Compute the rank-r DMD approximation from data snapshots X1, X2
    % X1: State snapshots at time 1, size (n, m)
    % X2: State snapshots at time 2, size (n, m)

    % Perform Singular Value Decomposition (SVD) on X1
    [U, S, V] = svd(X1, 'econ');

    % Truncate to rank-r
    r = rank(X1)
    U_r = U(:, 1:r);
    S_r = S(1:r, 1:r);
    V_r = V(:, 1:r);

    % Build Atilde and DMD modes
    Atilde = U_r' * X2 * V_r / S_r;
    [W, D] = eig(Atilde);

    % DMD modes
    dmd_results.Phi = X2 * V_r / S_r * W;

    % Eigenvalues
    dmd_results.Lambda = D;
end


