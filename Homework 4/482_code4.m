% TEST 1

file = 'Adele1.mp3';
[song1,Fs1]=audioread(file);
song1 = song1'/ 2;
song1 = song1(1,:) + song1(2,:);   
file = 'Adele2.mp3';
[song2,Fs2]=audioread(file);
song2 = song2'/ 2;
song2 = song2(1,:) + song2(2,:);
file = 'Adele3.mp3';
[song3,Fs3]=audioread(file);
song3 = song3'/ 2;
song3 = song3(1,:) + song3(2,:);


file = 'Adele4.mp3';
[song4,Fs4]=audioread(file);
song4 = song4'/ 2;
song4 = song4(1,:) + song4(2,:);

file = 'bach1.mp3';
[song5,Fs5]=audioread(file);
song5 = song5'/ 2;
song5 = song5(1,:) + song5(2,:);

file = 'bach2.mp3';
[song6,Fs6]=audioread(file);
song6 = song6'/ 2;
song6 = song6(1,:) + song6(2,:);

file = 'bach3.mp3';
[song7,Fs7]=audioread(file);
song7 = song7'/ 2;
song7 = song7(1,:) + song7(2,:);


A1 = [];
for kk = 40:5:160
    test = song1(1, Fs1*kk : Fs1*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A1 = [A1;vector];
end
A2 = [];
for kk = 40:5:160
    test = song2(1, Fs2*kk : Fs2*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A2 = [A2;vector];
end
A3 = [];
for kk = 40:5:160
    test = song3(1, Fs3*kk : Fs3*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A3 = [A3;vector];
end
A4 = [];
for kk = 40:5:160
    test = song4(1, Fs4*kk : Fs4*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A4 = [A4;vector];
end
A5 = [];
for kk = 40:5:160
    test = song5(1, Fs5*kk : Fs5*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A5 = [A5;vector];
end
A6 = [];
for kk = 40:5:160
    test = song7(1, Fs7*kk : Fs7*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A6 = [A6;vector];
end

A = cat(1,A1,A2,A3,A4,A5,A6);

A = A';


[u, s, v] = svd(A - mean(A(:)), 'econ');
plot(diag(s) ./ sum(diag(s)), 'ro');

thr = v';
plot3(thr(2, 1:50), thr(3, 1:50), thr(4, 1:50), 'ro'); hold on;
plot3(thr(2, 51:100), thr(3, 51:100), thr(4, 51:100), 'mo'); hold on;
plot3(thr(2, 101:150), thr(3, 101:150), thr(4, 101:150), 'co');
legend('Adele(Fast)', 'Adele(Slow)', 'Classic')
xlabel('Mode 2'); ylabel('Mode 3'); zlabel('Mode 4')


neighbors = []; naive = []; linear = [];
for test_trail = 1:1
    true = [ones(20,1); 2*ones(20,1); 3*ones(20,1)];
    q1 = randperm(50); q2 = randperm(50); q3 = randperm(50);
    x1 = v(1:50, 2:4);
    x2 = v(51:100, 2:4);
    x3 = v(101:150, 2:4);
    trainx = [x1(q1(1:30), :); x2(q2(1:30), :); x3(q3(1:30),:)];
    testx = [x1(q1(31:end), :); x2(q2(31:end), :); x3(q3(31:end),:)];
    % knn
    ind = knnsearch(trainx, testx); 
    for i = 1:length(ind)
       if ind(i) <= 30
           ind(i) =  1;
       elseif ind(i) <= 60
           ind(i) = 2;
       else
           ind(i) = 3;
       end
    end
    temp = [ind==true];
    neighbors(test_trail) = sum(temp) / length(temp);
    subplot(3,3,1)
    bar(ind);
    title('kNN');
    xlabel('Test data'); ylabel('Label');
    
    % naive bayes
    ctrain = [ones(30,1); 2*ones(30,1); 3*ones(30,1)];
    nb = fitcnb(trainx, ctrain);
    pre = nb.predict(testx);
    temp = [pre== true];
    naive(test_trail) = sum(temp) / length(temp);
    subplot(3,3,2);
    bar(pre)
    title('Naive Bayes');
    xlabel('Test data'); ylabel('Label');
    
    % classify (Built in)
    pre = classify(testx, trainx, ctrain);
    temp = [pre== true];
    linear(test_trail) = sum(temp) / length(temp);
    subplot(3,3,3);
    bar(pre);
    title('LDA');
    xlabel('Test data'); ylabel('Label');    
end

neighbors = mean(neighbors);
naive = mean(naive);
linear = mean(linear);

% TEST 2

file = 'Nirvana1.mp3';
[song1,Fs1]=audioread(file);
song1 = song1'/ 2;
song1 = song1(1,:) + song1(2,:);   
file = 'Nirvana2.mp3';
[song2,Fs2]=audioread(file);
song2 = song2'/ 2;
song2 = song2(1,:) + song2(2,:);
file = 'SG1.mp3';
[song3,Fs3]=audioread(file);
song3 = song3'/ 2;
song3 = song3(1,:) + song3(2,:);


file = 'SG2.mp3';
[song4,Fs4]=audioread(file);
song4 = song4'/ 2;
song4 = song4(1,:) + song4(2,:);

file = 'Alice1.mp3';
[song5,Fs5]=audioread(file);
song5 = song5'/ 2;
song5 = song5(1,:) + song5(2,:);

file = 'Alice2.mp3';
[song7,Fs7]=audioread(file);
song7 = song7'/ 2;
song7 = song7(1,:) + song7(2,:);


A1 = [];
for kk = 40:5:160
    test = song1(1, Fs1*kk : Fs1*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A1 = [A1;vector];
end
A2 = [];
for kk = 40:5:160
    test = song2(1, Fs2*kk : Fs2*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A2 = [A2;vector];
end
A3 = [];
for kk = 40:5:160
    test = song3(1, Fs3*kk : Fs3*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A3 = [A3;vector];
end
A4 = [];
for kk = 40:5:160
    test = song4(1, Fs4*kk : Fs4*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A4 = [A4;vector];
end
A5 = [];
for kk = 40:5:160
    test = song5(1, Fs5*kk : Fs5*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A5 = [A5;vector];
end
A6 = [];
for kk = 40:5:160
    test = song7(1, Fs7*kk : Fs7*(kk+5));
%   test = resample(test, 20000, Fs1);
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A6 = [A6;vector];
end

A = cat(1,A1,A2,A3,A4,A5,A6);

A = A';


[u, s, v] = svd(A - mean(A(:)), 'econ');
plot(diag(s) ./ sum(diag(s)), 'ro');

thr = v';
plot3(thr(2, 1:50), thr(3, 1:50), thr(4, 1:50), 'ro'); hold on;
plot3(thr(2, 51:100), thr(3, 51:100), thr(4, 51:100), 'mo'); hold on;
plot3(thr(2, 101:150), thr(3, 101:150), thr(4, 101:150), 'co');
legend('Nirvana', 'Soundgarden', 'Alice in Chains')
xlabel('Mode 2'); ylabel('Mode 3'); zlabel('Mode 4')


neighbors = []; naive = []; linear = [];
for test_trail = 1:1
    true = [ones(20,1); 2*ones(20,1); 3*ones(20,1)];
    q1 = randperm(50); q2 = randperm(50); q3 = randperm(50);
    x1 = v(1:50, 2:4);
    x2 = v(51:100, 2:4);
    x3 = v(101:150, 2:4);
    trainx = [x1(q1(1:30), :); x2(q2(1:30), :); x3(q3(1:30),:)];
    testx = [x1(q1(31:end), :); x2(q2(31:end), :); x3(q3(31:end),:)];
    % knn
    ind = knnsearch(trainx, testx); 
    for i = 1:length(ind)
       if ind(i) <= 30
           ind(i) =  1;
       elseif ind(i) <= 60
           ind(i) = 2;
       else
           ind(i) = 3;
       end
    end
    temp = [ind==true];
    neighbors(test_trail) = sum(temp) / length(temp);
    subplot(3,3,1)
    bar(ind);
    title('kNN');
    xlabel('Test data'); ylabel('Label');
    
    % naive bayes
    ctrain = [ones(30,1); 2*ones(30,1); 3*ones(30,1)];
    nb = fitcnb(trainx, ctrain);
    pre = nb.predict(testx);
    temp = [pre== true];
    naive(test_trail) = sum(temp) / length(temp);
    subplot(3,3,2);
    bar(pre)
    title('Naive Bayes');
    xlabel('Test data'); ylabel('Label');
    
    % classify (Built in)
    pre = classify(testx, trainx, ctrain);
    temp = [pre== true];
    linear(test_trail) = sum(temp) / length(temp);
    subplot(3,3,3);
    bar(pre);
    title('LDA');
    xlabel('Test data'); ylabel('Label');    
end

neighbors = mean(neighbors);
naive = mean(naive);
linear = mean(linear);

% TEST 3

file = 'Drake1.mp3';
[song1,Fs1]=audioread(file);
song1 = song1'/ 2;
song1 = song1(1,:) + song1(2,:);   
file = 'Drake2.mp3';
[song2,Fs2]=audioread(file);
song2 = song2'/ 2;
song2 = song2(1,:) + song2(2,:);
file = 'Eminem1.mp3';
[song3,Fs3]=audioread(file);
song3 = song3'/ 2;
song3 = song3(1,:) + song3(2,:);


file = 'Eminem2.mp3';
[song4,Fs4]=audioread(file);
song4 = song4'/ 2;
song4 = song4(1,:) + song4(2,:);

file = 'Kanye3.mp3';
[song5,Fs5]=audioread(file);
song5 = song5'/ 2;
song5 = song5(1,:) + song5(2,:);

file = 'Kanye2.mp3';
[song7,Fs7]=audioread(file);
song7 = song7'/ 2;
song7 = song7(1,:) + song7(2,:);


A1 = [];
for kk = 40:5:160
    test = song1(1, Fs1*kk : Fs1*(kk+5));
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A1 = [A1;vector];
end
A2 = [];
for kk = 40:5:160
    test = song2(1, Fs2*kk : Fs2*(kk+5));
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A2 = [A2;vector];
end
A3 = [];
for kk = 40:5:160
    test = song3(1, Fs3*kk : Fs3*(kk+5));
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A3 = [A3;vector];
end
A4 = [];
for kk = 40:5:160
    test = song4(1, Fs4*kk : Fs4*(kk+5));

    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A4 = [A4;vector];
end
A5 = [];
for kk = 40:5:160
    test = song5(1, Fs5*kk : Fs5*(kk+5));
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A5 = [A5;vector];
end
A6 = [];
for kk = 40:5:160
    test = song7(1, Fs7*kk : Fs7*(kk+5));
    vector = abs(spectrogram(test));
    vector = reshape(vector, [1, 8*32769]);
    A6 = [A6;vector];
end

A = cat(1,A1,A2,A3,A4,A5,A6);

A = A';


[u, s, v] = svd(A - mean(A(:)), 'econ');
plot(diag(s) ./ sum(diag(s)), 'ro');

thr = v';
plot3(thr(2, 1:50), thr(3, 1:50), thr(4, 1:50), 'ro'); hold on;
plot3(thr(2, 51:100), thr(3, 51:100), thr(4, 51:100), 'mo'); hold on;
plot3(thr(2, 101:150), thr(3, 101:150), thr(4, 101:150), 'co');
legend('Eminem', 'Drake', 'Kanye')
xlabel('Mode 2'); ylabel('Mode 3'); zlabel('Mode 4')


neighbors = []; naive = []; linear = [];
for test_trail = 1:1
    true = [ones(20,1); 2*ones(20,1); 3*ones(20,1)];
    q1 = randperm(50); q2 = randperm(50); q3 = randperm(50);
    x1 = v(1:50, 2:4);
    x2 = v(51:100, 2:4);
    x3 = v(101:150, 2:4);
    trainx = [x1(q1(1:30), :); x2(q2(1:30), :); x3(q3(1:30),:)];
    testx = [x1(q1(31:end), :); x2(q2(31:end), :); x3(q3(31:end),:)];

    ind = knnsearch(trainx, testx); 
    for i = 1:length(ind)
       if ind(i) <= 30
           ind(i) =  1;
       elseif ind(i) <= 60
           ind(i) = 2;
       else
           ind(i) = 3;
       end
    end
    temp = [ind==true];
    neighbors(test_trail) = sum(temp) / length(temp);
    subplot(3,3,1)
    bar(ind);
    title('kNN');
    xlabel('Test data'); ylabel('Label');
    

    ctrain = [ones(30,1); 2*ones(30,1); 3*ones(30,1)];
    nb = fitcnb(trainx, ctrain);
    pre = nb.predict(testx);
    temp = [pre== true];
    naive(test_trail) = sum(temp) / length(temp);
    subplot(3,3,2);
    bar(pre)
    title('Naive Bayes');
    xlabel('Test data'); ylabel('Label');
    
  
    pre = classify(testx, trainx, ctrain);
    temp = [pre== true];
    linear(test_trail) = sum(temp) / length(temp);
    subplot(3,3,3);
    bar(pre);
    title('LDA');
    xlabel('Test data'); ylabel('Label');    
end

neighbors = mean(neighbors);
naive = mean(naive);
linear = mean(linear);