#! /usr/bin/perl

use strict;
use warnings;

use Text::CSV_XS;
use File::Copy qw(copy);

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
