#! /usr/bin/perl 

use strict;
use warnings;

use Text::CSV_XS;
use Text::CSV::Slurp;
use Getopt::Long;
use Log::Log4perl qw( :easy );
Log::Log4perl->easy_init( $INFO );
use Try::Tiny;

GetOptions(
    'sample=s{1,}' => \my @raw_data,
    'output=s'     => \my $new_file,
    'tabs'         => \my $tabs,
    #'procs=i'      => \$procs,
    'verbose'      => \my $verbose,
    'ins'          => \my $ins,
    'dels'         => \my $dels
) or die("Command line argument error\n");

if (scalar @raw_data == 0){@raw_data = @ARGV}

OUTER: foreach my $current_file (@raw_data) {
#my $pid = $pm->start and next OUTER; # Forks and returns pid of c

    if ($ins) {
        $new_file = 'classification_neural_network_insertions_' . $current_file;
    } elsif ($dels) {
        $new_file = 'classification_neural_network_deletions_' . $current_file;
    } else  {
        $new_file = 'classification_neural_network_' . $current_file;
    }
	
# Only Miseq .. onwards has ref sequence in data due to crispresso update and have tab separators so don't need default csv
	#my $csv = Text::CSV_XS->new() or die Text::CSV_XS->error_diag();
	my $csv = Text::CSV_XS->new ({sep_char  =>  "\cI"}) or die Text::CSV_XS->error_diag();
	my @position;
	#if ($tabs) {$csv = $tsv}

	my @dataset;
	my $output_label;

	if ($ins) {
    	$output_label = 'insertions';
	} elsif ($dels) {
    	$output_label = 'deletions';
	} else {
    	$output_label = 'ins/dels';
	}

	open (my $data, '<', $current_file) or die "Could not find or open '$current_file'\n";

	my $headers = $csv->getline($data);

	my @dataset_header = $csv->column_names(@{$headers});
	
	for (my $x = 0;  my $observation = $csv->getline_hr($data); $x++) {

		my $ref = $observation->{'Reference_Sequence'};
		my $reads = $observation->{'#Reads'};
		my $insertions = $observation->{'n_inserted'};
		my $deletions = $observation->{'n_deleted'};
		

		if (substr($ref, 0, 1) eq '-'){$ref =~ s/^-//}	
		if (substr($ref, 0, -1) eq '-'){$ref =~ s/^-//}	
$DB::single=1;
		my @features = variables($ref, $insertions, $deletions);
		my $output = pop @features;	
		my $results = {};
		my $i = 0;
		
		foreach my $dinucleotide (@features) {
			$i++;
			$results->{$i} = $dinucleotide;
### Temporary fix ###
			if ($x == 0) {
###				
				push (@position, $i);
			}
		}
		$results->{$output_label} = $output;
$DB::single=1;
		push @dataset, ($results) x $reads;
		
### Temporary fix ###
		if ($x == 0) {
###
			push (@position, $output_label);
		}


	}
	close $data;

    open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
    try {
$DB::single=1;
        my $slurp = Text::CSV::Slurp->create(input => \@dataset, field_order => \@position);
        print $out $slurp;
        INFO "Data stored into new neural network file '$new_file'";
    } catch {
        warn "Caught error: $_";
    };
    close $out;

}


sub variables {

	my ($ref, $insertions, $deletions) = @_;
	my $output;
	
	my @features = ($ref =~ m/..?/g);
	
    if ($dels) {
        $output = $deletions;
    } elsif ($ins) {
        $output = $insertions;
    } else {
        $output = ($deletions*-1) + $insertions;
    }
	print "$output\n";

	return @features, $output;
	
}
