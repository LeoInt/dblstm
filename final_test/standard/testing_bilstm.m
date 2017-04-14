clear all
clc
%format long
%%%%%%%%%%%%%%%%%%%%%%%%% Init %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 seqLength=8;
 numLayers=4;
 hiddenSize=4;
 input_dim=2*hiddenSize;
 output_dim=6;
 if input_dim!=2*hiddenSize
  W1_i=zeros(4*hiddenSize,input_dim);
  W1=zeros(4*hiddenSize,2*hiddenSize,numLayers-1);
  W1_bw_i=zeros(4*hiddenSize,input_dim);
  W1_bw=zeros(4*hiddenSize,2*hiddenSize,numLayers-1);
 else
  W1=zeros(4*hiddenSize,2*hiddenSize,numLayers); 
  W1_bw=zeros(4*hiddenSize,2*hiddenSize,numLayers);
 endif
 %C_IN=zeros(hiddenSize,seqLength+1,numLayers);
 W2=zeros(4*hiddenSize,hiddenSize,numLayers);
 W2_bw=zeros(4*hiddenSize,hiddenSize,numLayers);
 B=zeros(4*hiddenSize,1,numLayers);
 B_bw=zeros(4*hiddenSize,1,numLayers);
 X=zeros(2*hiddenSize,seqLength,2);
 X_in=zeros(input_dim,seqLength);
 H=zeros(hiddenSize,seqLength+1);
 H_bw=zeros(hiddenSize,seqLength+1);
 C_IN=zeros(hiddenSize,seqLength+1);
 C_IN_bw=zeros(hiddenSize,seqLength+1);
 WA=zeros(output_dim,seqLength);
 bA=zeros(output_dim)
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I/O %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %X(:,:,1)=dlmread("X.txt");
 X_in(:,:)=dlmread("X.txt");
 %X=single(X);
 %%read the input from file, seqLength columns and 2*hiddensize rows 
 for layer = 1:numLayers  
  W1_name=strcat ("Wx",num2str(layer-1),".txt");
  W2_name=strcat ("Wh",num2str(layer-1),".txt");
  B_name=strcat ("bias",num2str(layer-1),".txt");
  if layer == 1 
    if input_dim!=2*hiddenSize
      W1_i(:,:)=dlmread(W1_name);
    else
      W1(:,:,layer)=dlmread(W1_name);
    endif
  else
    W1(:,:,layer)=dlmread(W1_name);
  endif

  W2(:,:,layer)=dlmread(W2_name);
  B(:,:,layer)=dlmread(B_name);
  B_name=strcat ("bias",num2str(layer-1),"bw.txt");  
  W1_name=strcat ("Wx",num2str(layer-1),"bw.txt");
  W2_name=strcat ("Wh",num2str(layer-1),"bw.txt");
  if layer == 1 
    if input_dim!=2*hiddenSize
      W1_bw_i(:,:)=dlmread(W1_name);
    else
      W1_bw(:,:,layer)=dlmread(W1_name);
    endif
  else
    W1_bw(:,:,layer)=dlmread(W1_name);
  endif
  W2_bw(:,:,layer)=dlmread(W2_name);
  B_bw(:,:,layer)=dlmread(B_name);
  p_name=strcat ("phole_i",num2str(layer-1),".txt");
  phole_i(:,:,layer)=dlmread(p_name);
  p_name=strcat ("phole_f",num2str(layer-1),".txt");
  phole_f(:,:,layer)=dlmread(p_name);
  p_name=strcat ("phole_o",num2str(layer-1),".txt");
  phole_o(:,:,layer)=dlmread(p_name);
  p_name=strcat ("phole_i",num2str(layer-1),"bw.txt");
  phole_i_bw(:,:,layer)=dlmread(p_name);
  p_name=strcat ("phole_f",num2str(layer-1),"bw.txt");
  phole_f_bw(:,:,layer)=dlmread(p_name);
  p_name=strcat ("phole_o",num2str(layer-1),"bw.txt");
  phole_o_bw(:,:,layer)=dlmread(p_name);
 endfor
WA=dlmread("WA.txt");
bA=dlmread("bA.txt");
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I/O %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 for layer = 1:numLayers
    if layer == 1 
      if input_dim!=2*hiddenSize
        Xt=W1_i(:,:)*X_in(:,:); %Xt will have T rows and 4*hiddensize cols
        Xt_bw=W1_bw_i(:,:)*X_in(:,:);
      else
        Xt=W1(:,:,layer)*X_in(:,:); %Xt will have T rows and 4*hiddensize cols
        Xt_bw=W1_bw(:,:,layer)*X_in(:,:);
      endif
    else
      Xt=W1(:,:,layer)*X(:,:,mod(layer-1,2)+1); %Xt will have T rows and 4*hiddensize cols
      Xt_bw=W1_bw(:,:,layer)*X(:,:,mod(layer-1,2)+1);
    endif
    for frame = 1:seqLength
      Xt1 = Xt(:,frame); %Xt1 will be a row
  %I1 = Tx1(1:1*hiddenSize);
  %I2 = Tx1(1*hiddenSize+1:2*hiddenSize);
  %I3 = Tx1(2*hiddenSize+1:3*hiddenSize);
  %I4 = Tx1(3*hiddenSize+1:4*hiddenSize);
      B1=B(1:4*hiddenSize,:,layer);
      Ht=W2(:,:,layer)*H(:,frame); %Ht will be a column
      
      Res=Ht+Xt1+B1;
      %piazzare il calcolo phole_f,i,o
      I2=tanh(Res(1:hiddenSize));
      Res(hiddenSize+1:2*hiddenSize)=Res(hiddenSize+1:2*hiddenSize).+ C_IN(:,frame).*phole_i(:,:,layer);
      I=sigmoid(Res(hiddenSize+1:2*hiddenSize));
      Res(2*hiddenSize+1:3*hiddenSize)=Res(2*hiddenSize+1:3*hiddenSize).+ C_IN(:,frame).*phole_f(:,:,layer);
      F=sigmoid(Res(2*hiddenSize+1:3*hiddenSize));
      %Res(3*hiddenSize+1:4*hiddenSize)=Res(3*hiddenSize+1:4*hiddenSize).+ C_IN(:,frame).*phole_o(:,:,layer);
      O=Res(3*hiddenSize+1:4*hiddenSize);
      F=F.*C_IN(:,frame);
      C=I.*I2;
      C=C.+F;
      C_IN(:,frame+1)=C;
      O=O.+(C.*phole_o(:,:,layer));
      O=sigmoid(O);
      result=O.*tanh(C);
      H(:,frame+1)=result;
      X(1:hiddenSize,frame,mod(layer,2)+1)=result;
      %X(1:hiddenSize,frame,mod(layer,2)+1)
    endfor
    for frame = seqLength:-1:1
      Xt1_bw = Xt_bw(:,frame); %Xt1 will be a row
      Ht_bw=W2_bw(:,:,layer)*H_bw(:,frame+1); %Ht will be a column
      B1_bw=B_bw(1:4*hiddenSize,:,layer);
      %Res_bw=Ht_bw+Xt1_bw+B1_bw+B2_bw;
      if frame == seqLength-5
        Res_bw=Ht_bw+Xt1_bw+B1_bw;
        test=Res_bw;
        h_test=Ht_bw;
      else
        Res_bw=Ht_bw+Xt1_bw+B1_bw;
      endif;
      %piazzare il calcolo phole_f,i,o
      %Res_bw(1:hiddenSize)=Res_bw(1:hiddenSize).+ C_IN_bw(:,frame+1).*phole_i_bw(:,:,layer);
      I2_bw=tanh(Res_bw(1:hiddenSize));
      Res_bw(hiddenSize+1:2*hiddenSize)=Res_bw(hiddenSize+1:2*hiddenSize).+ C_IN_bw(:,frame+1).*phole_i_bw(:,:,layer);
      I_bw=sigmoid(Res_bw(hiddenSize+1:2*hiddenSize));
      Res_bw(2*hiddenSize+1:3*hiddenSize)=Res_bw(2*hiddenSize+1:3*hiddenSize).+ C_IN_bw(:,frame+1).*phole_f_bw(:,:,layer);
      F_bw=sigmoid(Res_bw(2*hiddenSize+1:3*hiddenSize));
      %Res_bw(3*hiddenSize+1:4*hiddenSize)=Res_bw(3*hiddenSize+1:4*hiddenSize).+ C_IN_bw(:,frame+1).*phole_o_bw(:,:,layer);
      O_bw=Res_bw(3*hiddenSize+1:4*hiddenSize);
      
      F_bw=F_bw.*C_IN_bw(:,frame+1);
      C_bw=I_bw.*I2_bw;
      C_bw=C_bw.+F_bw;
      C_IN_bw(:,frame)=C_bw;
      O_bw=O_bw.+(C_bw.*phole_o_bw(:,:,layer));
      %H_bw(:,frame)=tanh(I_bw);
      O_bw=sigmoid(O_bw);
      result_bw=O_bw.*tanh(C_bw)
      if frame == seqLength-4
        restest=result_bw;
      endif
      H_bw(:,frame)=result_bw;
      X(hiddenSize+1:2*hiddenSize,frame,mod(layer,2)+1)=result_bw;
      %X(hiddenSize+1:2*hiddenSize,frame,mod(layer,2)+1)     
     endfor
 endfor
out=WA*X(:,:,mod(layer,2)+1);
for frame = 1:seqLength
  out(:,frame)+=bA;
endfor
%y = e^x_j/sum_j(e^x_j)
outsoft=zeros(output_dim,seqLength)
for frame = 1:seqLength
  m=max(out(:,frame));
  s=0;
  for i = 1:output_dim
    outsoft(i,frame)=e^(out(i,frame)-m);
    s+=outsoft(i,frame);
  endfor
  for i = 1:output_dim
    outsoft(i,frame)=outsoft(i,frame)/s;
  endfor
endfor