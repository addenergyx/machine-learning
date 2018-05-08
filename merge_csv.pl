#! /usr/bin/perl

# For classifcation must merge data before running preprocessing script, this is because each file would contain headers of different lengths due to varying sequence sizes. The classifaction script handles this and uses the longest sequence size as the header.
# For regression run script on the data first then merge the files
 

use strict;
use warnings;

my @file;
my $headers;

foreach my $file (@ARGV) {
	
	# Add new line to end of file
	#`echo >> $file`;

	open my $data, '<', $file or die $!;
	
	my @lines = <$data>;
	
	$headers = shift @lines;
	push (@file, @lines);
	close $data;
	
	# Remove new line after
	#`perl -i -ne 'print unless eof' $file`
}

unshift (@file, $headers);
open my $out, '>', 'Merged.csv' or die $!;
print $out @file;
close $out;
