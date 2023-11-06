function process_trd(filelist, root_dir)

fileID = fopen(filelist);
files = textscan(fileID,'%s');
fclose(fileID);


numfiles = numel(files{1});
mydata = cell(1, numfiles);
for k = 1:numfiles
  myfilename = char(files{1}{k});
  myfilename
  process_wheel(myfilename);
end