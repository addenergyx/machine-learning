#! /usr/bin/perl

use strict;
use warnings;

use Text::CSV_XS;
use Text::CSV::Slurp;
use Getopt::Long;
use Data::Dumper;

GetOptions(
	'tabs'			=>	\my $tabs,
	'files=s{1,}'	=>	\my @files,
);

my $csv = Text::CSV_XS->new or die Text::CSV_XS->error_diag;
my $tsv = Text::CSV_XS->new ({sep_char	=>	"\cI"}) or die Text::CSV_XS->error_diag;

if ($tabs){$csv = $tsv}
my @dataset;

if (scalar @files == 0){@files = @ARGV}

foreach my $file (@files) {
	#shift @files;
	open (my $fh, '<', $file) or die "'$file':$!\n";
	my $headers = $csv->getline($fh);
	$csv->column_names(@{$headers});
	while (my $row = $csv->getline_hr($fh)) {
		delete $row->{'Reference_Sequence'};
		push (@dataset, $row);
#$DB::single=1;
		print "\nRow " . scalar @dataset . " in file $file has been modified\n" ;
	}
	close $fh;

	open (my $out, '>', $file) or die "'$file'";
$DB::single=1;
	my $element = 'Reference_Sequence';
	@{$headers} = grep {$_ ne $element} @{$headers};
	#splice @{$headers},1,1;
	my $slurp = Text::CSV::Slurp->create(input=>\@dataset, field_order => \@{$headers});
	print $out $slurp;

	close $out;
	print "Closed $file";
	undef (@dataset);
	# undef(@array) frees all the memory associated with the array except the bare minimum
}

