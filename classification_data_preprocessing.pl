#! /usr/bin/perl 

use strict;
use warnings;

use Text::CSV_XS;
use Text::CSV::Slurp;
use Getopt::Long;
use Log::Log4perl qw( :easy );
Log::Log4perl->easy_init( $INFO );
use Try::Tiny;
use Parallel::ForkManager;

my $procs = `nproc --all`;

GetOptions(
    'sample=s{1,}' => \my @raw_data,
    'output=s'     => \my $new_file,
    'procs=i'      => \$procs,
	'ins'          => \my $ins,
    'dels'         => \my $dels,
) or die("Command line argument error\n");

my $pm = Parallel::ForkManager->new($procs);

if (scalar @raw_data == 0){@raw_data = @ARGV}

OUTER: foreach my $current_file (@raw_data) {
    my $pid = $pm->start and next OUTER;
	
    if (! defined $new_file) {
    	if ($ins) {
       		$new_file = 'classification_52bp_neural_network_insertions_' . $current_file;
   		} elsif ($dels) {
       		$new_file = 'classification_52bp_neural_network_deletions_' . $current_file;
   		} else {
       		$new_file = 'classification_52bp_neural_network_' . $current_file;
    	}
	}

# Only Miseq 19 onwards has ref sequence in data due to crispresso update and are tab separated
# Don't need forking because the csvs' are merged into one csv (7798000+ results) before running this script.
# Some results have double quotes in them so needed to change quote and escape char defaults so that the result were not taken as one string
	my $csv = Text::CSV_XS->new ({sep_char  =>  "\cI", quote_char => "?", escape_char => "?"}) or die Text::CSV_XS->error_diag();

    # Miseq_Plate_Crispr_Map.csv contains all the crisprs relating to each experiment
	my $crispr_map_csv = Text::CSV_XS->new() or die Text::CSV_XS->error_diag();
	my $crispr;
	
	# Gets Experiment name from filename, Experiment names are unique
	my ($exp) = $current_file =~ /(?<=exp)(.*)(?=_Alleles)/gi;
	open (my $crispr_map, '<', '/home/ubuntu/data-preprocessing-worktree/data/Miseq_Plate_Crispr_Map.csv') or die "Can not find Miseq_Plate_Crispr_Map.csv";
	my $map_headers = $crispr_map_csv->getline($crispr_map);
	
	my @crispr_header = $crispr_map_csv->column_names(@{$map_headers});

	MAPPER: while (my $row = $crispr_map_csv->getline_hr($crispr_map)) {
		my $experiment = $row->{'Experiment'};
		if (lc($exp) eq lc($experiment)) {
			$crispr = $row->{'CRISPR'};
			INFO "Found crispr '$crispr' related to experiment $experiment from Miseq_Plate_Crispr_Map.csv";
			last MAPPER;
		}
        # If no matching crispr is found need to undefine crispr
        undef $crispr;
	}
	
	close $crispr_map;

### FIX STDIN doesn't work with forking, keeps looping ### 	
#	if (! defined $crispr) {
#		print "No crispr relating to experiment $exp found, please add experiment and crispr to Miseq_Plate_Crispr_Map.csv or continue and script will search database for an appropriate crispr";
#		print "\nPress <Enter> to continue or Ctrl-C to quit: ";
#		my $input = <STDIN>;
#		$pm->finish and goto OUTER if lc($input) eq "q";
#	}
###

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
	
	ROW: for (my $x = 1;  my $observation = $csv->getline_hr($data); $x++) {
		
		my $ref = $observation->{'Reference_Sequence'};
		my $reads = $observation->{'#Reads'};
		my $insertions = $observation->{'n_inserted'};
		my $deletions = $observation->{'n_deleted'};

		my @features = variables($ref, $insertions, $deletions, $crispr, $x);
		
		unless (@features) { next ROW }
		
		my $output = pop @features;	
		my $results;
		my $i = 0;
		
		foreach my $dinucleotide (@features) {
			$i++;
			$results->{$i} = $dinucleotide;
			push (@position, $i);			
		}
		$results->{$output_label} = $output;
		for (my $y = 0; $y < $reads; $y++) {push @dataset, $results}

		push (@list_of_positions, [@position]);

		undef @position;
		$results = {};
	}
	
	# Only necessary when printing out sequences of different lengths
	#INFO "Searching data for longest sequence";
	foreach my $row (@list_of_positions) {
		if (scalar @{$row} > scalar @position) {@position = @{$row}}	
	}

	if (scalar @position > 0 ) {		
		push (@position, $output_label);
		close $data;

    	open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
    	try {
			INFO "Writing data into file $new_file";
        	my $slurp = Text::CSV::Slurp->create(input => \@dataset, field_order => \@position);
        	print $out $slurp;
        	INFO "Data stored into new neural network file '$new_file'";
    	} catch {
        	warn "Caught error: $_";
    	};
    	close $out;
	} else { 
		print "\nNo data created from file $current_file\n"; 
	}
	undef $new_file;
    $pm->finish;
}
$pm->wait_all_children;

sub variables {

	my ($ref, $insertions, $deletions, $crispr, $x) = @_;
	my $output;
	my $crispr_start;
	# '-' in reference sequence is an insertion whereas in the alligned sequence it is a deletion
	# To get the actual seq the lab put through sequencing '-' must be removed from the ref seq 
	$ref =~ s/-//g;
	
	unless(! defined $crispr) { $crispr_start = index(lc($ref), lc($crispr)) }
	# Sometimes the wrong crispr can be given for an experiment. If so must search through all 
	# crisprs in csv for a match. If no matches are found the observation is dropped. Most if not all crisprs for Miseq 19 are wrong

	if (! defined $crispr_start || $crispr_start == -1 ) {

		my $crispr_map_csv = Text::CSV_XS->new() or die Text::CSV_XS->error_diag();
	
		open (my $crispr_map, '<', '/home/ubuntu/data-preprocessing-worktree/data/Miseq_Plate_Crispr_Map.csv') or die "Can not find Crispr mapping CSV";
		my $map_headers = $crispr_map_csv->getline($crispr_map);
	
		my @crispr_header = $crispr_map_csv->column_names(@{$map_headers});

		RECOVERY: while (my $row = $crispr_map_csv->getline_hr($crispr_map)) {
			$crispr = $row->{'CRISPR'};
			if (index(lc($ref), lc($crispr)) != -1) {
				#print "CRISPR '$crispr' from experiment $row->{'Experiment'} is found in the reference sequence\n";
				$crispr_start = index(lc($ref), lc($crispr));
				last RECOVERY; 
			} else {
				undef $crispr;
			}	 
		}
		close $crispr_map;
	}
	
	# Bad idea to try to alter the control of a loop externally like this
	#if (! defined $crispr) { next ROW }

	# Drop row if can't find a crispr for reference sequence
	if (! defined $crispr) { 
		#print "Could not find crispr in reference sequence from database, dropping row: $x\n";
		return 
	}
	
	my $crispr_length = length($crispr);
	my $crispr_end = $crispr_start + $crispr_length - 1;

	# +3 includes pam
	my $size = $crispr_length + 3;
	
	# Neural network only takes 50 bases around crispr, crispr size should be 20 but if not this
	# will ensure only 50 bases are returned
	my $start = $crispr_start - (50 - $size);	

	# +3 to include pam
	my $site = substr $ref, $start, 52;

	my @features = ($site =~ m/..?/g);
	
    if ($dels) {
        $output = $deletions;
    } elsif ($ins) {
        $output = $insertions;
    } else {
        $output = ($deletions*-1) + $insertions;
    }
	
	#$crispr = lc($crispr);
	#print "$crispr\n$crispr_start\n$crispr_length\n$ref\n";
	#print "52 base pairs: $site\n";

	return (@features, $output);
	
}

