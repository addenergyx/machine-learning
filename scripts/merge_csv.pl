#! /usr/bin/perl

# For the classifcation dataset containing sequences of different lenghts must merge data before running preprocessing script, this is because each file would contain headers of different lengths due to varying sequence sizes. 
# The classifaction script handles this and uses the longest sequence size as the header.
# For regression dataset and classification when the number of bases is fixed (52bp) run scripts on each file individually before merging 
 
use strict;
use warnings;

my @file;
my $headers;

foreach my $file (@ARGV) {
	
	# Add new line to end of file
	# This is so the last result in one file doesn't merge with the beginning of another  
	`echo >> $file`;

	open my $data, '<', $file or die $!;
	
	my @lines = <$data>;
	
	$headers = shift @lines;
	push (@file, @lines);
	close $data;

	# Remove new line after
	#`perl -i -ne 'print unless eof' $file`
	print "Stored file $file\n"; 	
}

my $filename = 'miseq_classification_data.csv'; 
unshift (@file, $headers);
open my $out, '>', $filename or die $!;
print "Copying files to output\n";
print $out @file;
close $out;
