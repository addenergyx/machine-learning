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
	
# Only Miseq 19 onwards has ref sequence in data due to crispresso update and have tab separators so don't need default csv
# Won't need forking because the csvs' are merged into one csv (7798000+ results) before running this script.
# Some results have double quotes in then so needed to change quote and escape char defaults so that the result wasn't taken as one string
	my $csv = Text::CSV_XS->new ({sep_char  =>  "\cI", quote_char => "?", escape_char => "?"}) or die Text::CSV_XS->error_diag();
	my @position;
	my @list_of_positions; 
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
		
$DB::single=1;
		my $ref = $observation->{'Reference_Sequence'};
		my $reads = $observation->{'#Reads'};
		my $insertions = $observation->{'n_inserted'};
		my $deletions = $observation->{'n_deleted'};

		my @features = variables($ref, $insertions, $deletions);
		my $output = pop @features;	
		my $results;
		my $i = 0;
		
		foreach my $dinucleotide (@features) {
			$i++;
			$results->{$i} = $dinucleotide;
### Temporary fix ###
###if ($x == 0) {
###				
			push (@position, $i);			
###}
		}
		$results->{$output_label} = $output;
$DB::single=1;
		for (my $y = 0; $y < $reads; $y++) {push @dataset, $results}

		push (@list_of_positions, [@position]);
### Temporary fix ###
#if ($x == 0) {
###
#			push (@position, $output_label);
#}
		undef @position;
		$results = {};
		print "$x\n";
	}
$DB::single=1;	
	foreach my $row (@list_of_positions) {
		if (scalar @{$row} > scalar @position) {@position = @{$row}}	
	}

	push (@position, $output_label);
	close $data;

    open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
    try {
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

	# '-' in reference sequence is an insertion whereas in the alligned sequence it is a deletion
	# To get the actual seq the lab put through sequencing must remove the - 
	$ref =~ s/-//g;	
	my @features = ($ref =~ m/..?/g);
	
    if ($dels) {
        $output = $deletions;
    } elsif ($ins) {
        $output = $insertions;
    } else {
        $output = ($deletions*-1) + $insertions;
    }
	#print "$output\n";

	return @features, $output;
	
}
