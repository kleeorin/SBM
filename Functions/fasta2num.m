function [align]=fasta2num(file)

[xseqs]=fastaread(file);
xaln=char(xseqs.Sequence);
letter2number_map = create_letter2number_map();
align=zeros(size(xaln));
for index=1:size(xaln,1)                           %loop over all sequences in the alignment 
      
    
	align(index,:) = ...                   %append converted sequence to output alignments
    letter2number_map(xaln(index,:)); 
    
    
end

end
function letter2number_map = create_letter2number_map()
%characters are indexed by decimal ASCII codes (which range from 0 to 255, with - ->45 and A ->65 etc.)
letter2number_map(256) = 0; %initialize all bytes to 0 (1 line and 256 cols of 0s)
letter2number_map('-') = 1;
letter2number_map('A') = 2;
letter2number_map('C') = 3;
letter2number_map('D') = 4;
letter2number_map('E') = 5;
letter2number_map('F') = 6;
letter2number_map('G') = 7;
letter2number_map('H') = 8;
letter2number_map('I') = 9;
letter2number_map('K') = 10;
letter2number_map('L') = 11;
letter2number_map('M') = 12;
letter2number_map('N') = 13;
letter2number_map('P') = 14;
letter2number_map('Q') = 15;
letter2number_map('R') = 16;
letter2number_map('S') = 17;
letter2number_map('T') = 18;
letter2number_map('V') = 19;
letter2number_map('W') = 20;
letter2number_map('Y') = 21;
letter2number_map('B') = -1; %ambiguous : skip sequences containing these
letter2number_map('Z') = -1; %ambiguous : skip sequences containing these
letter2number_map('J') = -1; %ambiguous : skip sequences containing these
letter2number_map('X') = -1; %ambiguous : skip sequences containing these
letter2number_map('U') = -1; %non-standard : skip sequences containing these
letter2number_map('O') = -1; %non-standard : skip sequences containing these
letter2number_map('a') = -2; %non-conserved: skip in seq of interest
letter2number_map('c') = -2; %non-conserved: skip in seq of interest
letter2number_map('d') = -2; %non-conserved: skip in seq of interest
letter2number_map('e') = -2; %non-conserved: skip in seq of interest
letter2number_map('f') = -2; %non-conserved: skip in seq of interest
letter2number_map('g') = -2; %non-conserved: skip in seq of interest
letter2number_map('h') = -2; %non-conserved: skip in seq of interest
letter2number_map('i') = -2; %non-conserved: skip in seq of interest
letter2number_map('k') = -2; %non-conserved: skip in seq of interest
letter2number_map('l') = -2; %non-conserved: skip in seq of interest
letter2number_map('m') = -2; %non-conserved: skip in seq of interest
letter2number_map('n') = -2; %non-conserved: skip in seq of interest
letter2number_map('p') = -2; %non-conserved: skip in seq of interest
letter2number_map('q') = -2; %non-conserved: skip in seq of interest
letter2number_map('r') = -2; %non-conserved: skip in seq of interest
letter2number_map('s') = -2; %non-conserved: skip in seq of interest
letter2number_map('t') = -2; %non-conserved: skip in seq of interest
letter2number_map('v') = -2; %non-conserved: skip in seq of interest
letter2number_map('w') = -2; %non-conserved: skip in seq of interest
letter2number_map('y') = -2; %non-conserved: skip in seq of interest
letter2number_map('b') = -2; %non-conserved: skip in seq of interest
letter2number_map('z') = -2; %non-conserved: skip in seq of interest
letter2number_map('j') = -2; %non-conserved: skip in seq of interest
letter2number_map('x') = -2; %non-conserved: skip in seq of interest
letter2number_map('u') = -2; %non-conserved: skip in seq of interest
letter2number_map('o') = -2; %non-conserved: skip in seq of interest
letter2number_map('.') = -3; %non-conserved: skip in seq of interest, do not advance position
end