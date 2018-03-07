#! /usr/bin/perl

use strict;
use warnings;

use Text::CSV_XS;
use Text::CSV::Slurp;
use Getopt::Long;
use Tie::File;

my $csv = Text::CSV_XS->new or die Text::CSV_XS->error_diag;
my @dataset;

foreach my $file (@ARGV) {

	open (my $fh, '<', $file) or die "'$file':$!\n";
	my $headers = $csv->getline($fh);
	$csv->column_names(@{$headers});
		
	while (my $row = $csv->getline_hr($fh)){
		delete $row->{'Reference_Sequence'};
		push (@dataset, $row);
	}
close $fh;

open (my $out, '>', $file) or die "'$file'";
splice @{$headers},1,1;
my $slurp = Text::CSV::Slurp->create(input=>\@dataset, field_order => \@{$headers});
print $out $slurp;

close $out;
}

