%% Clear Workspace and console
clear; 
clc;


%% Import Data

load('train.mat');
[n, m] = size(X);

%Convert class 0 to class -1
for i = 1:n
    if y(i) ~= 1
        y(i) = -1;
    end
end


%% Running/Testing

%K for K-Fold
k = 4;

%Beginning of test fold
test_begin = 1;

%Regularisation Parameter C
c = 0.1;

%Split the data into training data and testing data
[X_train, y_train, X_test, y_test] = cross_validate(X, y, k, test_begin);

%Train the model
model = svm_train_dual(X_train, y_train, c);

%Test the model
acc = svm_predict_dual(X_test, y_test, model);

disp("All tests complete, Total accuracy = " + acc*100 + "%");


%% Cross Validate Data

function [X_train, y_train, X_test, y_test] = cross_validate(X, y, k, te_start)
    [n, m] = size(X); 
    c = n / k;

    te_end = mod(te_start + c - 1, n);

    if te_start == 1
        X_train = X(te_end+1:end,:);
        y_train = y(:, te_end+1:end);
        X_test = X(te_start:te_end,:);
        y_test = y(:,te_start:te_end);
    else
        if (n - te_start) < n / k  
            X_train = X(te_end+1:te_start-1,:);
            y_train = y(:,te_end+1:te_start-1);       

            X_test = [ X(1:te_end,:); X(te_start:end,:) ];
            y_test = [ y(: ,1:te_end), y(:,te_start:end) ];
        else
            X_train = [ X(1:te_start-1,:); X(te_end+1:end,:) ];
            y_train = [ y(:,1:te_start-1), y(:,te_end+1:end)];

            X_test = X(te_start:te_end,:);
            y_test = y(:,te_start:te_end);
        end
    end


    if (n - te_start) < n / k  
        X_test = [ X(1:te_end,:); X(te_start:end,:) ];
        y_test = [ y(: ,1:te_end), y(:,te_start:end) ];
    else 
        X_test = X(te_start:te_end,:);
        y_test = y(:,te_start:te_end);
    end
end


%% Primal Train

function svm_model = svm_train_primal(data_train , label_train , regularisation_para_C)
    
    [n, m] = size(data_train);

    cvx_begin quiet
        cvx_precision low
        variables slack(n) w(m) b
        minimize (0.5 * (w' * w) + ((regularisation_para_C/n) * sum(slack)))
        subject to
            label_train' .* (data_train * w + b) >= 1 - slack;
            slack >= 0;
    cvx_end

    svm_model.w = w;
    svm_model.b = b;
    
end


%% Primal Testing

function test_accuracy = svm_predict_primal(data_test , label_test , svm_model)
    
    incorrect = 0;
    n = length(label_test);

    for i = 1:n
        if label_test(i) == 0
            if ( dot(svm_model.w, data_test(i,:)) + svm_model.b ) >= 0
                incorrect = incorrect + 1;
            end
        elseif label_test(i) == 1
            if ( dot(svm_model.w, data_test(i,:)) + svm_model.b ) < 0
                incorrect = incorrect + 1;
            end
        end
    end
    
    test_accuracy = (n - incorrect) / n;
end


%% Dual Training

function svm_model_d = svm_train_dual( data_train , label_train , regularisation_para_C)
    
    n = size(data_train, 1);
    
    %Linear Kernal
    linK = data_train * data_train';
    
    cvx_begin quiet
    cvx_precision low
    variable al(n)
    minimize ( 0.5 .* quad_form(label_train' .* al, linK) - ones(n, 1)' * al);
    subject to
        0 <= al;
        al <= regularisation_para_C / n;
        label_train * al == 0;
    cvx_end
    
    w = (al .* label_train')' * data_train;
    b = mean(label_train' - w * data_train', 'all');
    
    svm_model_d.w = w;
    svm_model_d.b = b;
    
end


%% Dual Testing

function test_accuracy_d = svm_predict_dual(data_test , label_test , svm_model_d)
    
    incorrect = 0;
    n = length(label_test);

    for i = 1:n
        if label_test(i) == 0
            if ( dot(svm_model_d.w, data_test(i,:)) + svm_model_d.b ) >= 0
                incorrect = incorrect + 1;
            end
        elseif label_test(i) == 1
            if ( dot(svm_model_d.w, data_test(i,:)) + svm_model_d.b ) < 0
                incorrect = incorrect + 1;
            end
        end
    end
    
    test_accuracy_d = (n - incorrect) / n;
    
end



